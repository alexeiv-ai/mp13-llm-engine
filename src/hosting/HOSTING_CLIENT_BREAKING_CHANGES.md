# Hosting Client Change Notes: Callable Surface Primitives

Date: 2026-06-21

This slice adds reusable Host Capability + Toolbox callable-surface helpers and removes implicit service-owned `fs.*` / `http.fetch` registration from workflow node dispatch. Clients must register callable sessions explicitly when sandbox code expects those methods.

## Client Actions

- Use `hosting.callable_surface.toolbox_to_host_capability_descriptors(...)` to convert toolbox describe output plus an optional `ToolsView` into Host Capability descriptors.
- Use `hosting.callable_surface.host_capability_descriptors_to_callable_schemas(...)` when a model-facing or sandbox-facing callable schema list is needed.
- Register or replace Host Capability sessions through `EngineHostControlChannel.host_capability_session_upsert(...)` instead of manually closing and registering sessions.
- Use `host_capability_session_list_filtered(...)` and `host_capability_session_close_filtered(...)` for workflow/instance/request/consumer/provider/owner/method scoped lifecycle operations.
- Use `bind_host_capability_provider_callback(...)`, `host_capability_provider_success(...)`, and `host_capability_provider_error(...)` to hide raw `hosting.sandbox.host_capability_call.v1` response handling.
- Use `HostCapabilityProviderCallbackRelay.bind_callback(...)` when registering a local `client_session` provider. Pass the returned binding to `host_capability_session_register(...)` or `host_capability_session_upsert(...)`.
- Use `host_capability_approval_request(...)` and `host_capability_approval_decision(...)` for approval bridge payloads.
- Approval decisions now have concrete broker semantics:
  - `deny` rejects the current call.
  - `allow_once` approves only the current call.
  - `add_to_scope` creates a scoped grant reused for later matching calls in the same broker/request context.
- Use `host_capability_audit_list(...)` for filtered Host Capability audit reads instead of parsing merged control state.
- Use `host_capability_session_register_toolbox(...)` to register hosted toolbox descriptors as a `toolbox_session` provider. The registered methods now execute through the existing toolbox harness.

## Client-Owned Callback Sessions

For local client-owned providers, create a callback relay and register its binding with the Host Capability session:

```python
from hosting import HostCapabilityProviderCallbackRelay

relay = HostCapabilityProviderCallbackRelay()
binding = relay.bind_callback(
    lambda method, arguments, context: {
        "customer_id": arguments["customer_id"],
        "request_id": context.get("request_id"),
    }
)

channel.host_capability_session_upsert(
    session_id="crm-provider",
    scope={"workflow_id": "wf-1"},
    visibility="workflow",
    methods=[
        {
            "name": "crm.customer.lookup",
            "namespace": "crm",
            "group_path": ["CRM", "Customer"],
            "args_schema": {"type": "object"},
            "result_schema": {"type": "object"},
        }
    ],
    binding=binding,
)
```

The relay validates `provider_call_id` and returns normalized success/error envelopes. Release the relay binding when the provider session is closed:

```python
relay.release(binding)
```

## Built-In Migration Path

- Service-owned `fs.*` / `http.fetch` fallback is no longer enabled by default.
- Clients should register known broker-supported methods with `host_capability_session_register_known_methods(...)` or custom client-owned implementations.
- A diagnostic fallback can be explicitly enabled per request policy with:

```json
{
  "sandbox": {
    "host_api": {
      "service_owned_fallback_enabled": true
    }
  }
}
```

- When the fallback is used, hosting now records `host_capability_service_fallback_used` audit rows and emits a warning log event.
- Treat the fallback as diagnostics only; do not rely on it for normal workflow execution.

## Correlation

Safe correlation metadata is preserved by helper APIs for:

`workflow_id`, `instance_id`, `node_id`, `request_id`, `cursor_id`, `context_id`, `branch_id`, `session_tree_id`, `actor`, `provider_id`, `method`, `approval_id`, `host_call_id`, and `provider_call_id`.

## Host-Managed State Capabilities

Workflow Python node requests can now opt into host-managed state methods:

- `state.workflow.get`, `state.workflow.set`, `state.workflow.list`, `state.workflow.delete`
- `state.instance.*` when the request has an `instance_id`
- `state.backend.*` only when explicitly enabled with `state.backend=true`

State is disabled by default. Enable workflow/instance state with:

```json
{
  "sandbox": {
    "host_api": {
      "state": true
    }
  }
}
```

Enable backend-global state only with an explicit scoped policy:

```json
{
  "sandbox": {
    "host_api": {
      "state": {
        "workflow": true,
        "instance": true,
        "backend": true
      }
    }
  }
}
```

Sandbox code should call these through `host.call(...)`:

```python
host.call("state.workflow.set", {"key": "customer.profile", "value": {"tier": 2}})
profile = host.call("state.workflow.get", {"key": "customer.profile"})
```

Writes are versioned. Pass `expected_version` when the client wants optimistic conflict detection; mismatches raise `state_version_conflict`.

## Python Node Pinned Instances

Clients can now create and route requests through explicit Python node module/snippet instances:

- `workflow_python_instance_create(...)`
- `workflow_python_instance_execute(...)`
- `workflow_python_instance_list()`
- `workflow_python_instance_close(...)`

Use this when process-local mutation is intentional and the client wants later calls to hit the same live worker process:

```python
created = channel.workflow_python_instance_create(instance_id="inst-1", request=template_request)
out = channel.workflow_python_instance_execute(instance_id="inst-1", request=run_request)
channel.workflow_python_instance_close(instance_id="inst-1")
```

Current limits:

- Python project mode is rejected for pinned instances until cwd, `sys.path`, env, and import-cache policy is explicit.
- Pinned instance restart recovery is not implemented yet; use explicit host-managed state for data that must survive close/restart.

## JavaScript Node Pinned Instances

Clients can now create and route requests through explicit JavaScript node module/snippet instances:

- `workflow_js_instance_create(...)`
- `workflow_js_instance_execute(...)`
- `workflow_js_instance_list()`
- `workflow_js_instance_close(...)`

Use this when the client wants later calls to hit the same live QuickJS worker process:

```python
created = channel.workflow_js_instance_create(instance_id="js-inst-1", request=template_request)
out = channel.workflow_js_instance_execute(instance_id="js-inst-1", request=run_request)
channel.workflow_js_instance_close(instance_id="js-inst-1")
```

Current limits:

- JS project mode is rejected for pinned instances until cwd, env, module/cache, and cleanup policy is explicit.
- The worker process is pinned, but each JS request still creates a fresh QuickJS context. Use host-managed state for data that must persist between calls.
- Pinned instance restart recovery is not implemented yet.

## Host-Managed State Snapshots

Clients can now snapshot and restore explicit host-managed state partitions:

- `sandbox_state_snapshot(scope=..., workflow_id=..., instance_id=..., request_id=..., prefix=...)`
- `sandbox_state_restore(snapshot=..., scope=..., workflow_id=..., instance_id=..., request_id=..., mode="merge"|"replace")`

Use this for restart recovery of `state.instance.*` data:

```python
snapshot = channel.sandbox_state_snapshot(scope="instance", workflow_id="wf-1", instance_id="inst-1")
channel.sandbox_state_restore(snapshot=snapshot, workflow_id="wf-1", instance_id="inst-2", mode="replace")
```

Snapshots contain only explicit host-managed state. They do not serialize Python globals, QuickJS contexts, import caches, cwd, environment mutations, or arbitrary worker memory.
