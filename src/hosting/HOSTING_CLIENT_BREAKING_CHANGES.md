# Hosting Client Breaking Changes

Date: 2026-06-21

Scope: deferred cleanup from the completed Sandbox Event Stream Protocol pillar.

## Workflow Event Read Command Rename

- Removed public workflow stream receive commands:
  - `workflow-python-stream-recv`
  - `workflow-js-stream-recv`
- Use the event subscription commands instead:
  - `workflow-python-event-subscribe`
  - `workflow-js-event-subscribe`

Client-channel helpers now expose:

```python
channel.workflow_python_event_subscribe(stream_id=stream_id, max_items=64)
channel.workflow_js_event_subscribe(stream_id=stream_id, max_items=64)
```

The removed channel helpers are:

```python
channel.workflow_python_stream_recv(...)
channel.workflow_js_stream_recv(...)
```

## Subscription Response Shape

`workflow-*-event-subscribe` returns the event batch contract and helper-normalized events:

```json
{
  "status": "ok",
  "stream_id": "...",
  "request_id": "...",
  "batch": {
    "version": 1,
    "context": {},
    "base": {},
    "loss": {"output": 0, "event": 0, "audit": 0},
    "frames": [],
    "more": false
  },
  "normalized_events": [],
  "closed": false,
  "canceled": false
}
```

Do not depend on legacy workflow recv fields:

- `events`
- `max_events`
- `retained_event_count`
- `dropped_event_count`
- `next_sequence`

Use `batch.loss` or the helper `stream_loss` normalized event to detect loss.

## Auth And CLI Changes

- RBAC/authorization allowlists no longer include `workflow-python-stream-recv` or `workflow-js-stream-recv`.
- CLI command tables no longer expose `workflow-python-stream-recv` or `workflow-js-stream-recv`.
- Interactive workflow runtime event viewing now calls `workflow-*-event-subscribe` and displays normalized events.

## Payload Notes

- Normalized events use `kind` instead of legacy `type`.
- Event fields are flattened in normalized events instead of nested under legacy `payload`.
- Artifact events use event `kind="artifact"`; artifact metadata such as `ref`, `filename`, `media_type`, and `size_bytes` are top-level normalized fields.

## Not Changed In This Slice

- `workflow-*-stream-open`, `workflow-*-stream-send`, and `workflow-*-stream-close` remain public workflow commands.
- Low-level proxy/generic-worker stream commands such as `proxy-stream-recv` are not changed in this slice.
- Built-in sandbox `host.call(...)` behavior is not changed in this slice.

## Host Capability Discovery

Date: 2026-06-21

Scope: first Host Capability Protocol implementation slice.

### New Discovery API

Python workflow nodes can now call:

```python
described = sandbox.describe()
```

JavaScript workflow nodes can now call:

```javascript
const described = sandbox.describe();
```

The returned discovery document uses:

```json
{
  "contract": "hosting.sandbox.discovery.v1",
  "harness": {},
  "events": {},
  "host_capabilities": {
    "methods": [],
    "groups": [],
    "providers": [],
    "transport": {}
  },
  "state": {},
  "actions": {},
  "policy": {},
  "roots": {}
}
```

### `host.describe()` / `api.describe()` Additions

`host.describe()` and JavaScript `api.describe()` now return the same discovery-oriented shape while preserving top-level `methods`, `method_descriptions`, `policy`, `roots`, and `transport` for current callers.

Important changes for clients:

- `methods` now includes `sandbox.describe`.
- `method_descriptions` include `group_path` and sandbox-safe `provider` metadata.
- `host_capabilities.methods` contains shared capability descriptors for built-ins such as `fs.*` and `http.fetch`.
- Provider discovery intentionally omits callback binding addresses and provider session tokens.

Clients that only check for method presence should continue using `methods`. Clients that need schemas, scopes, groups, providers, or future client-owned capabilities should switch to `host_capabilities.methods`.

## Host Capability Provider Sessions

Date: 2026-06-21

Scope: provider session lifecycle API slice.

### New Daemon Commands

The daemon now accepts authenticated control-scope commands:

- `host-capability-session-register`
- `host-capability-session-list`
- `host-capability-session-close`

Normal clients should prefer the hosting channel helpers:

```python
channel.host_capability_session_register(
    session_id="crm-provider",
    visibility="workflow",
    scope={"workflow_id": "wf-1"},
    methods=[
        {
            "name": "crm.customer.lookup",
            "group_path": ["CRM", "Customer"],
            "args_schema": {"type": "object"},
            "result_schema": {"type": "object"},
        }
    ],
)
channel.host_capability_session_list()
channel.host_capability_session_close(session_id="crm-provider")
```

### Public Response Shape

Public session responses expose descriptor metadata and lifetime fields, but they do not expose provider callback bindings, callback addresses, or provider session tokens.

Provider callback invocation is not public yet in this slice. Registered client-owned methods are lifecycle-managed and discoverable by the daemon registry work, but sandbox calls still only execute built-in providers until the provider callback RPC slice lands.

## Host Capability Provider Callback Envelope

Date: 2026-06-21

Scope: provider callback envelope and response-validation slice.

Provider calls now use the canonical internal contract `hosting.sandbox.host_capability_call.v1` with `provider_call_id`, `method`, `arguments`, and `context`.

Provider responses must echo the exact `provider_call_id` and use either:

```json
{"status": "ok", "provider_call_id": "...", "result": {}}
```

or:

```json
{"status": "error", "provider_call_id": "...", "reason": "provider_reason", "message": "", "detail": {}}
```

Normal clients should still use hosting library helpers once callback transport is exposed; they should not manually construct low-level callback envelopes unless using raw daemon commands intentionally.

## Host Capability Provider Timeout And Cancellation

Date: 2026-06-21

Scope: provider callback timeout, disconnect, and cancellation slice.

Structured provider failures now preserve their reason through Python and JavaScript node host responses:

- `host_call_timeout` for provider callback timeout
- `host_capability_provider_unavailable` for missing/disconnected provider transport
- `host_call_canceled` for broker/request cancellation

Clients that surface sandbox host-call errors should prefer the returned `reason` field when it is available instead of parsing the error message string.
