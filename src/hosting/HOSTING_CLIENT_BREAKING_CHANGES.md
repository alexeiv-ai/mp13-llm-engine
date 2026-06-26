# Hosting Client Breaking Changes

## Toolbox Per-IO Host API Approval Is Publicly Forwarded

The previously worker-only `host_api_approval` option is now exposed through the
public toolbox execution path.

Client action:

1. When toolbox `context.fs.*`, `context.http.*`, or `context.host.call(...)`
   should require per-IO Host Capability approval, pass
   `host_api_approval={"mode": "always"}` or another Host Capability approval
   policy on toolbox execution.
2. Public entrypoints that now forward `host_api_approval`:
   `EngineHostControlChannel.toolbox_execute(...)`,
   `EngineHostService.toolbox_execute(...)`,
   `HostedToolBoxRef.execute(...)`, and
   `ToolboxExecutionHarness.execute_calls(...)` /
   `ToolboxExecutionHarness.execute_request_tools(...)`.
3. Keep passing the hosted callback processor/binding as before. Per-IO approval
   requests are delivered over that existing callback binding using the Host
   Capability approval callback name and
   `hosting.sandbox.host_capability_approval.v1` payload shape.

This does not replace tool-level gated execution. Tool gating still controls
whether a tool can run. `host_api_approval` controls brokered host API calls made
from inside an already-running toolbox tool.
