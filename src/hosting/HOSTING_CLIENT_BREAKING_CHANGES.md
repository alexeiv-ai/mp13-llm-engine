# Hosting Client Change Notes: Callable Surface Primitives

Date: 2026-06-21

This slice adds reusable Host Capability + Toolbox callable-surface helpers and removes implicit service-owned `fs.*` / `http.fetch` registration from workflow node dispatch. Clients must register callable sessions explicitly when sandbox code expects those methods.

## Client Actions

- Use `hosting.callable_surface.toolbox_to_host_capability_descriptors(...)` to convert toolbox describe output plus an optional `ToolsView` into Host Capability descriptors.
- Use `hosting.callable_surface.host_capability_descriptors_to_callable_schemas(...)` when a model-facing or sandbox-facing callable schema list is needed.
- Register or replace Host Capability sessions through `EngineHostControlChannel.host_capability_session_upsert(...)` instead of manually closing and registering sessions.
- Use `host_capability_session_list_filtered(...)` and `host_capability_session_close_filtered(...)` for workflow/instance/request/consumer/provider/owner/method scoped lifecycle operations.
- Use `bind_host_capability_provider_callback(...)`, `host_capability_provider_success(...)`, and `host_capability_provider_error(...)` to hide raw `hosting.sandbox.host_capability_call.v1` response handling.
- Use `host_capability_approval_request(...)` and `host_capability_approval_decision(...)` for approval bridge payloads.
- Use `host_capability_audit_list(...)` for filtered Host Capability audit reads instead of parsing merged control state.
- Use `host_capability_session_register_toolbox(...)` to register hosted toolbox descriptors as a `toolbox_session` provider. The registered methods now execute through the existing toolbox harness.

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
