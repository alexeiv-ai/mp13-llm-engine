# Hosting Client Breaking Changes

Date: 2026-06-22

## Callable Surface Bridge Integration Instructions

No wire-level breaking change is pending for this slice. The following helper-level integration rules are available for clients that expose multiple hosted toolbox instances, Host Capability providers, or merged sandbox/model callable views.

### Native Toolbox Metadata Boundary

Toolbox metadata remains toolbox-owned. Host Capability descriptors are an adapter/export format, not native toolbox storage.

Use `toolbox_to_callable_schemas(...)` when the client needs a sandbox/model-facing callable list directly from toolbox metadata. It preserves:

- stable namespace-qualified method names;
- hierarchical `group_path`;
- visibility and gated/disabled state from `ToolsView`;
- provider identity;
- schema, method, and policy digests;
- toolbox metadata such as original tool name, toolbox id, view id, mode, and constraints.

### Merged Callable Views

Multiple provider sessions may expose the same bare method name. Do not merge them by name only.

Use the callable-surface helpers so each advertised method carries:

- `identity.provider_kind`
- `identity.provider_id`
- `identity.toolbox_id`
- `identity.session_id`
- `identity.method`
- `schema_digest`
- `method_digest`
- `policy_digest`

`host_capability_descriptors_to_callable_schemas(...)` now rejects duplicate advertised callable names by default with `callable_surface_duplicate_name:<name>`. Resolve conflicts by using provider-specific namespaces/aliases before advertising the merged surface, or pass `conflict_policy="keep_first"` only when the caller explicitly wants first-provider wins behavior.

For concurrent toolbox instances with overlapping tool names, prefer provider-specific namespaces such as `crm_a.lookup` and `crm_b.lookup` while preserving provider/session identity in the callable schema.

### Host-Side API Providers

Do not model every host-side API provider as a real toolbox. A host-side API should be a generic callable provider unless it actually needs toolbox lifecycle, install/config, storage, and toolbox execution semantics.

Use toolbox-shaped metadata for descriptors, schemas, visibility, approval policy, constraints, and callback wiring. Use a real toolbox only for providers that are operationally toolbox instances.

### Approval Scope

Approval grants default to the same provider/session scope. Cross-toolbox or cross-session reuse must be explicit and should include compatible:

- owner/workspace scope
- toolbox definition digest
- `schema_digest`
- `method_digest`
- `policy_digest`
- method name and provider kind
- approved scope constraints

Clients continue to own workflow-facing durable approval state. Hosting owns live broker decisions, optional broker-scoped grants, and durable audit.

### Explicit Bridge Policy

Brokered IO permissions should use an explicit bridge policy. Do not rely on implicit inheritance from toolbox policy or Host Capability policy alone.

Use `host_capability_bridge_policy(...)` to compute and record the effective namespace intersection across:

- toolbox-side policy
- Host Capability caller policy
- explicit bridge policy

Missing bridge-policy namespace entries deny access by default.

### Correlation

Approval/audit/callback payloads should preserve safe correlation metadata when available:

- `workflow_id`
- `instance_id`
- `node_id`
- `request_id`
- `cursor_id`
- `context_id`
- `branch_id`
- `session_tree_id`
- `session_id`
- `toolbox_id`
- `actor`
- `provider_kind`
- `provider_id`
- `method`
- `approval_id`
- `host_call_id`
- `provider_call_id`
