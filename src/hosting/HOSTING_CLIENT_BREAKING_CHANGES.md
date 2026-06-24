# Hosting Client Adoption Notes

Date: 2026-06-23

## Dynamic Sandbox Action Discovery

This is an additive feature. Existing static `action_manifest` / `actions`
requests and default `run(payload)` execution continue to work.

Clients that want sandbox-owned card/action discovery should adopt the new
public action-describe parameters:

1. Call `workflow_python_action_describe(dynamic=True, request=...)` or
   `workflow_js_action_describe(dynamic=True, request=...)`.
2. Put the discovery entrypoint in
   `request["action_discovery"]["entrypoint"]` when the default is not enough.
   The default is `{"kind": "export", "export_name": "describe_actions"}`.
3. The sandbox discovery callable should return the normal action manifest
   shape, for example `{"output": {"actions": [...]}}` or
   `{"output": {"action_manifest": {...}}}`.
4. Reuse the returned manifest as `request["action_manifest"]` when invoking a
   selected action with `execute_workflow_python_action(...)` or
   `execute_workflow_js_action(...)`.
5. To discover actions against warm state, create the pinned instance first and
   pass the same `instance_id` to `workflow_*_action_describe(...)`. Action
   describe does not create instances implicitly.
6. If discovery needs Host Capability approvals, pass the same
   `approval_requester_binding` used by normal workflow execute calls.

Example Python request:

```python
request = {
    "request_id": "req-actions",
    "module_source": source,
    "module_sha256": source_sha,
    "package_id": "pkg",
    "workflow_id": "wf",
    "package_source_digest": "sha256:pkg",
    "payload": {"selection": "card-1"},
    "action_discovery": {
        "entrypoint": {"kind": "export", "export_name": "describe_actions"}
    },
}

manifest = channel.workflow_python_action_describe(
    dynamic=True,
    request=request,
    profile="node",
)

result = channel.execute_workflow_python_action(
    profile="node",
    action_name="preview",
    request={**request, "action_manifest": manifest},
)
```

Example JavaScript sandbox:

```javascript
exports.describe_actions = function(payload) {
  return {
    output: {
      actions: [
        {
          name: "preview",
          title: "Preview",
          entrypoint: { kind: "export", export_name: "preview" }
        }
      ]
    }
  };
};

exports.preview = function(payload) {
  return { output: { ok: true, payload } };
};
```
