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

SSH-bound sessions use normal hosting auth behavior. When a control session is bound to SSH, raw daemon calls for `host-capability-session-register`, `host-capability-session-list`, and `host-capability-session-close` must present the matching `_ssh_session_binding`; hosting library channel helpers attach this automatically for SSH targets.

### Public Response Shape

Public session responses expose descriptor metadata and lifetime fields, but they do not expose provider callback bindings, callback addresses, or provider session tokens.

Provider callback invocation uses the internal provider callback envelope described below. Normal clients should use hosting library helpers rather than constructing raw provider callback envelopes directly.

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

## Host Capability Scope And Precedence

Date: 2026-06-21

Scope: broker scope, namespace, permission, and duplicate-resolution slice.

Client-owned provider sessions must include scope fields matching their declared visibility:

- `visibility="request"` requires `scope.request_id`
- `visibility="workflow"` requires `scope.workflow_id`
- `visibility="instance"` requires `scope.instance_id`
- `visibility="consumer"` requires `scope.consumer_id`

Sessions outside the current broker scope are omitted from discovery and cannot be called. Duplicate method names resolve deterministically: built-ins win by default, then narrower client scopes win (`request`, `instance`, `workflow`, `consumer`), then session ID is the tie-breaker.

## Host Capability Approval Flow

Date: 2026-06-21

Scope: broker-level gated approval flow slice.

Descriptors with `approval.mode` other than `none` now require an outward approval decision before provider execution. Denial is surfaced as structured reason `host_call_approval_denied`, and the provider callback is not invoked.

Approval requests use internal contract `hosting.sandbox.host_capability_approval.v1` and include method, arguments, context, approval metadata, and sandbox-safe provider metadata. Provider callback bindings and provider session tokens are not included.

Approval decisions are also written to durable host capability audit state.

## Host Capability Event Observations

Date: 2026-06-21

Scope: broker event observations for host API calls.

Workflow event subscribers can now observe broker-generated host API events in addition to worker-generated `host_call` events:

- `host_response` for successful and failed broker dispatch
- `approval` for approval requested, approved, or denied
- `provider_failure` for provider timeout, disconnect, validation, or provider error
- `canceled` for canceled in-flight provider calls

Events include `method` and correlation fields. When the worker supplied a `host_call_id`, broker events expose it as both `host_call_id` and `call_id`; provider-backed calls also include `provider_call_id`. Client code should correlate on `call_id` for stream UX and use `provider_call_id` only for provider callback/debug flows.

Hosting library helpers should hide this wire shape for normal clients. Raw stream consumers should treat these events as observations, not as the sandbox-visible host response protocol itself.

## Host Capability Durable Approval Audit

Date: 2026-06-21

Scope: durable audit for gated host capability approvals.

Gated host capability approval outcomes are now persisted in hosting control state under `audit/host_capability_audit.json`. Records are decision-bearing and include approval/provider/call correlation IDs, method name, request/workflow/package context, provider metadata, approval metadata, argument key names, and sanitized decision details.

Provider bindings, callback addresses, provider session tokens, and raw argument values are not written to these audit records.

Clients using hosting library helpers should not need to parse this file directly. Raw clients that inspect hosting audit state should expect the new `host_capability_audit_events` bucket when reading merged control state.

## Host API Public Contract Completion

Date: 2026-06-21

Scope: Host API pillar completion.

The Host API pillar now exposes shared capability descriptors, sandbox discovery, brokered built-in dispatch, client-owned provider session lifecycle APIs, provider callback envelopes, permission/scope gates, approval routing, live event observations, and durable approval audit records.

Recommended client path: use the hosting library helpers for session registration, provider callback handling, stream observations, and audit reads. Raw daemon clients are responsible for auth tokens, SSH binding presentation, provider response validation, and correlation IDs.
