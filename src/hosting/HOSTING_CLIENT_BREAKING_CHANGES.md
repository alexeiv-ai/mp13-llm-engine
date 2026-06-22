# Hosting Client Breaking Changes

Date: 2026-06-22

## Diagnostic Service-Owned `fs.*` / `http.fetch` Fallback Removed

The workflow node Host API no longer supports the diagnostic service-owned fallback provider for:

- `fs.list`
- `fs.read_text`
- `fs.write_text`
- `fs.mkdir`
- `fs.stat`
- `http.fetch`

Setting `sandbox.host_api.service_owned_fallback_enabled=true`, `service_fallback_enabled=true`, or `service_owned_fallback=true` no longer registers these methods. Those policy keys are ignored by workflow node Host Capability dispatch.

### Required Client Action

Clients that want sandbox code to call `host.call("fs.*")`, Python convenience wrappers such as `host.fs_read_text(...)`, JavaScript convenience wrappers such as `api.fs.readText(...)`, or `host.call("http.fetch")` must register those methods through explicit Host Capability sessions before executing workflow nodes.

Use the high-level hosting client helpers where possible:

```python
channel.host_capability_session_register_known_methods(
    session_id="workflow-host-api",
    scope={"workflow_id": workflow_id},
    binding={
        "transport": "local_ipc",
        "callback_binding": callback_binding,
    },
    include_fs=True,
    include_http=True,
    allow_override=True,
)
```

For local callback providers, bind callbacks through `HostCapabilityProviderCallbackRelay` or an equivalent client-owned callback transport. The callback must handle the canonical provider call envelope `hosting.sandbox.host_capability_call.v1` and return a normalized provider response with the matching `provider_call_id`.

### Method Ownership

`api.fs` / `host.fs_*` / `api.http` convenience wrappers remain available in worker harnesses, but they are only convenience callers for advertised Host Capability methods. They do not imply that `fs.*` or `http.fetch` exists.

Clients may:

- register the standard known method descriptors with client-owned implementations;
- omit any methods they do not want sandbox code to discover;
- replace a known method with a custom implementation by registering the same fully-qualified method name with explicit override behavior.

Duplicate fully-qualified method registration still fails by default unless override is explicit.

### Expected Failure Without Registration

If a workflow node calls one of these methods without a matching Host Capability session, the call fails through normal host-call error handling with an unsupported method reason such as:

```text
unsupported_host_method:fs.read_text
unsupported_host_method:http.fetch
```

Sandbox discovery also omits unregistered methods from `host.describe()` / `sandbox.describe()`.

### Removed Diagnostics

The following diagnostic-only fallback behavior is removed:

- automatic service-owned fallback registration when `service_owned_fallback_enabled=true`;
- `host_capability_service_fallback_used` audit rows;
- warning log events saying `service-owned host capability fallback used`;
- fallback-only tests and docs that treated the hosting service as owner of `fs.*` / `http.fetch`.

Use Host Capability audit reads for client-owned provider calls instead.
