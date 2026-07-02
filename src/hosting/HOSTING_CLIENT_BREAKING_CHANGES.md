# Hosting Client Breaking Changes

## Host Capability Approval Requests Include `argument_preview`

The normalized `hosting.sandbox.host_capability_approval.v1` request now exposes
bounded sanitized argument values:

```python
{
    "method": "fs.read_text",
    "argument_keys": ["relative_path", "root_id"],
    "argument_preview": {
        "root_id": "project_ro",
        "relative_path": "src/app/mp13chat.py",
    },
}
```

Client action:

1. Update `host_capability.approval` callbacks to read policy-relevant values
   from `request["argument_preview"]`.
2. Do not read raw `request["arguments"]`; public approval callbacks should not
   depend on transport-specific raw argument payloads.
3. Continue using `argument_keys` only for display, diagnostics, and quick
   field-presence checks.
4. Treat `argument_preview` as approval-decision input, not final authority.
   Daemon sandbox policy and brokered IO enforcement remain authoritative.

Preview rules:

- small scalar values are preserved;
- secret-like keys such as `secret`, `token`, `password`, `api_key`, and
  `authorization` are redacted;
- large strings are summarized instead of copied;
- complex objects are summarized by type/key metadata.

## Approval Helper Functions

Clients may use the new helpers exported from `hosting`:

```python
from hosting import (
    host_capability_approval_check_fs_path,
    host_capability_approval_check_http_fetch,
)
```

Use `host_capability_approval_check_fs_path(...)` to validate
`root_id + relative_path` from `argument_preview` against a sandbox policy and
an optional scoped virtual root before returning `allow_once`.

Use `host_capability_approval_check_http_fetch(...)` to validate `http.fetch`
URL/method previews against brokered HTTP sandbox policy.

## Model-Facing Tool Schemas

Do not expose real filesystem roots or virtual root selectors as model-facing
tool arguments unless the tool intentionally supports a validated subroot
request. Host/client configuration owns real roots. Approval scope owns virtual
root narrowing. Model/tool calls should usually provide only relative targets
such as `relative_path`.
