# Hosting Client Breaking Changes

## Current Status

The dependent client has consumed the toolbox `host_api_approval` migration:

- hosted toolbox execution derives and forwards `host_api_approval`;
- connector execution forwards it through `ToolboxExecutionHarness.execute_calls(...)`;
- Host Capability approvals are handled separately from tool-level confirmation;
- source-tool `sandbox_policy` metadata is preserved at deploy time.

No additional public API migration is required for clients that execute hosted
tools through `EngineHostControlChannel.toolbox_execute(...)`,
`EngineHostService.toolbox_execute(...)`, `HostedToolBoxRef.execute(...)`, or
`ToolboxExecutionHarness`.

## Current Client Guidance

Toolbox worker calls to `context.fs.*`, `context.http.*`, and
`context.host.call(...)` are now dispatched back through the parent host service.
This keeps approval, audit, and broker policy enforcement parent-owned.

Client code should keep using the public toolbox execution APIs above and keep
passing `host_api_approval` plus the normal hosted callback binding. The
`host_capability.dispatch` callback is an internal parent-owned bridge; clients
should not register, inspect, or handle it directly.

If a client bypasses the public toolbox execution path and calls the toolbox
worker RPC directly, it must stop doing that for host API calls. Direct worker
RPC host calls no longer have enough context unless the parent-owned dispatch
binding is installed by the service layer.
