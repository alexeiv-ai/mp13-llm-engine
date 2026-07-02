# Hosting Client Breaking Changes

No mandatory client breaking changes.

## Optional Host Capability Approval Helper Adoption

The hosting library now exposes registry-owned service-broker policy hints:

```python
from hosting import (
    host_capability_approval_check_service_broker_request,
    service_broker_method_policy_hint,
)
```

Clients may replace local method-name maps such as `fs.read_text -> read` and
`fs.write_text -> write` with:

```python
check = host_capability_approval_check_service_broker_request(
    request,
    sandbox_policy,
    scoped_root=approved_scoped_root,
)
```

For lower-level dispatch, use:

```python
hint = service_broker_method_policy_hint(request["method"])
```

Current hints cover `fs.*` and `http.fetch`. Future daemon-owned brokered
methods should add their policy hints in the service-broker registry so client
approval code does not drift.
