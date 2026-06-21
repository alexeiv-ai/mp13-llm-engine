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
