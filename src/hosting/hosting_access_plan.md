---

# Feature request: Real bounded concurrency for hosted toolbox and Host API calls

## Summary

Please add end-to-end concurrent execution support for hosted toolbox calls in the parent hosting component.

The toolbox execution harness already schedules calls concurrently with `asyncio.gather`, but production calls made through `EngineHostControlChannel` are serialized by the persistent connection lock. As a result, clients cannot obtain real parallel tool execution even when they submit calls concurrently.

Our downstream project plans to execute tool calls from different batch response items concurrently by default. It needs the parent hosting layer to provide bounded, observable, and safe concurrency rather than merely accepting concurrent client tasks that serialize internally.

This request does not require changing parent chat’s current batch policy. Parent chat may continue to serialize its own batch tool application; the hosting API should permit concurrency for clients that request it.

## Current behavior

1. `ToolboxExecutionHarness.execute_calls(..., parallel=True)` schedules calls with `asyncio.gather`.
2. Sandbox calls use `asyncio.to_thread(control_channel.toolbox_execute, ...)`.
3. `LocalSocketConnection.invoke()` and `SSHRelayConnection.invoke()` hold a connection lock for the complete request/response exchange.
4. The daemon invokes synchronous `svc.toolbox_execute(...)` directly inside its async dispatcher, blocking the daemon event loop.
5. The hosted process pool tracks logical capacity but:

   - normally retains one toolbox executor worker slot;
   - does not queue saturated requests;
   - returns `capacity_exceeded` immediately;
   - does not load-balance multiple registrations for the same tool.

6. The toolbox executor process can handle separate IPC connections in separate threads, but the upstream control and daemon paths prevent normal clients from reaching it concurrently.

Therefore the public harness appears concurrent while the production transport remains effectively serial.

## Requested behavior

### 1. Concurrent control transport

Support multiple in-flight `toolbox-execute` requests from one client channel.

Possible implementations include:

- a multiplexed persistent connection with request IDs and independent responses;
- a bounded pool of persistent control connections;
- one short-lived connection per concurrent execution request.

The solution must work for both local and supported remote/SSH control transports without response interleaving.

### 2. Non-blocking daemon dispatch

Tool execution must not block the daemon event loop.

Please either:

- provide an async toolbox execution service; or
- dispatch synchronous execution through a bounded executor using `asyncio.to_thread` or equivalent.

Control operations such as cancellation and status inspection must remain responsive while tools are running.

### 3. Atomic capacity management

Make toolbox pool admission thread-safe and atomic.

Concurrent submissions must not overbook a worker because two callers observe the same available slot. Likewise, completion and cancellation must release exactly one admitted slot.

### 4. Bounded queue or explicit backpressure

Please define a real saturation policy rather than relying only on immediate `capacity_exceeded`.

Preferred behavior:

- configurable maximum active calls;
- configurable bounded queue depth;
- queue wait timeout;
- cancellation while queued;
- explicit errors such as `queue_full` and `queue_timeout`;
- queue and execution timing in request diagnostics.

Fail-fast admission may remain an available policy, but the effective policy should be discoverable.

### 5. Tool concurrency policy

Add optional concurrency metadata to tool definitions, for example:

```json
{
  "concurrency": {
    "mode": "parallel",
    "group": "",
    "max_concurrency": 8
  }
}
```

Supported modes should include:

- `parallel`: may overlap with other calls;
- `serial`: only one call at a time;
- `keyed`: calls sharing a derived resource key are serialized;
- optionally `exclusive`: blocks other calls in the same toolbox.

This is needed for tools that mutate the same file, browser session, workspace, database, or external resource.

If no metadata is provided, the hosting team should define and document the compatibility default. Our downstream scheduler intends to request concurrent execution by default while preserving serial handling for known control and mutation operations.

### 6. All-settled execution results

A failure in one concurrently executing call must not discard successful sibling results.

Please return one stable result for every admitted call, including:

- tool call ID;
- result or error;
- queued/start/finish timestamps;
- cancellation status;
- worker identity;
- concurrency or serialization decision;
- retry/admission information.

Unexpected execution exceptions should be normalized into per-call failures.

### 7. Cancellation

Cancellation should work independently for every queued or running call.

Canceling one call must not recycle an entire toolbox executor unless coarse worker termination is unavoidable. If worker recycling is required, all affected sibling calls must receive explicit `sandbox_recycled` results.

### 8. Host API concurrency contract

Please clarify and expose concurrency behavior for Host Capability/Host API callbacks.

The callback relay can already accept connections on separate threads, but there is no general per-provider/per-method concurrency contract.

Requested additions:

- provider/method concurrency metadata;
- optional serial or keyed execution policy;
- maximum in-flight callback count;
- bounded callback queue/backpressure;
- cancellation and timeout behavior;
- documentation that provider callbacks must be thread-safe when marked parallel.

Backend-owned Host API providers are workflow-facing capabilities, not automatically ordinary model tools. This separation should remain intact.

### 9. Capability discovery and diagnostics

`toolbox.describe` or an equivalent endpoint should report effective runtime behavior, for example:

```json
{
  "parallel_execution": {
    "supported": true,
    "effective_max_concurrency": 8,
    "queue_policy": "bounded",
    "queue_depth": 32,
    "queue_timeout_seconds": 30,
    "worker_process_count": 1,
    "execution_model": "threaded_worker"
  }
}
```

Please distinguish:

- logical call capacity;
- actual worker process count;
- thread/task concurrency within a worker;
- queue depth;
- active and queued calls.

Calling the current structure a worker pool can otherwise imply multiple worker processes when toolbox execution actually uses one registered executor with logical capacity.

## Acceptance criteria

1. Two hosted tools that each block for one second complete in approximately one second, not two, when invoked concurrently through the real production path:

```text
client harness
→ EngineHostControlChannel
→ daemon
→ toolbox runtime admission
→ toolbox executor IPC
→ actual tool functions
```

2. The test must use the real control transport and daemon. A fake channel is insufficient.

3. The same test passes for:

   - two calls in one tool round;
   - calls from two independent batch branches;
   - two independent client requests.

4. A serial-marked tool remains strictly serialized.

5. Keyed calls with the same resource key serialize, while different keys may overlap.

6. Saturation produces deterministic queueing or documented fail-fast behavior without overbooking.

7. Canceling one running or queued call does not silently lose sibling results.

8. Status and cancellation commands remain responsive while long tools execute.

9. Metrics accurately report active, queued, completed, failed, timed-out, and canceled calls.

10. Host API callback concurrency has equivalent overlap, saturation, and cancellation coverage.

## Downstream integration

Once this is available, our project will:

1. Record batch assistant/tool-call turns deterministically.
2. Flatten eligible calls across batch response items.
3. Submit them concurrently with a bounded limit.
4. Wait for an all-settled round barrier.
5. Record results in stable batch-item and tool-call order.
6. Start the next inference round only after the barrier completes.

The downstream project will keep these operations serialized unless explicitly allowed:

- `toolbox_search_and_scope`;
- approval decisions;
- tool-scope mutations;
- calls sharing a protected resource/concurrency key.

## Provider identity

Grok uses an OpenAI-compatible transport but is a distinct provider. Any shared Responses/OpenAI-compatible hosting or diagnostics contract should keep these fields separate:

```json
{
  "provider_id": "grok",
  "transport_adapter": "responses_api",
  "protocol_compatibility": "openai"
}
```

Shared protocol handling must not rewrite the canonical provider identity to `openai`.

Anthropic support is outside the scope of this request.

## Non-goals

- Changing parent chat’s current serialized batch-tool policy.
- Automatically exposing workflow Host API providers as ordinary model tools.
- Parallelizing successive inference/tool rounds; each round should retain its barrier.
- Guaranteeing that every existing tool is safe for concurrent mutation.
- Requiring multiple processes when bounded concurrency in one worker is sufficient.

## Priority and rationale

This is needed before downstream batch tool concurrency can provide actual latency improvement. Without parent changes, adding further `asyncio.gather` calls downstream only creates tasks that wait behind the production control-channel lock.
