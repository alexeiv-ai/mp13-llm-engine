# Reply to the dependent project: hosted concurrency contract

Date: 2026-08-02

The requested feature is now implemented through the production hosting path:
`EngineHostControlChannel` → daemon → hosted runtime admission → toolbox
executor IPC. Toolbox execution requests use independent local or SSH request
transports, daemon service calls are dispatched off the event loop, and the
runtime pool now has atomic admission, bounded waiting, policy gates, request
lifecycle diagnostics, and explicit cancellation outcomes.

## What to stop doing

- Stop assuming that `asyncio.gather` alone guarantees overlap. Submit eligible
  calls through the hosting channel and honor the returned admission and
  diagnostics fields.
- Stop treating `capacity_exceeded` as the only saturation result. Handle
  `queue_full` and `queue_timeout` as deterministic per-call failures, and do
  not retry either blindly.
- Stop depending on response arrival order. Keep the original tool-call ID and
  restore the downstream batch-item/tool-call order at the all-settled round
  barrier.
- Stop canceling a whole toolbox to remove one queued call. Cancel by stable
  `tool_call_id`; a queued request is removed independently. If a running call
  requires coarse worker termination, sibling requests are returned with the
  explicit `sandbox_recycled` reason.
- Stop interpreting `worker_process_count` as the call-concurrency limit. Read
  `logical_call_capacity`, `active_calls`, `queued_calls`, and
  `execution_model` separately from the process count.
- Stop assuming every tool is safe to mutate concurrently. The compatibility
  default is parallel, so mutation tools must declare a serial or keyed policy.
- Stop exposing workflow-facing Host API providers as ordinary model tools.
  Host API discovery and model-tool discovery remain separate surfaces.
- Stop rewriting provider identity when a protocol is compatible. A provider
  such as Grok must retain its canonical `provider_id` and may separately
  report protocol compatibility (for example, `openai`).

## What to start doing

- Start flattening eligible calls, submitting them with a bounded client-side
  limit, awaiting an all-settled result set, and preserving the existing round
  barrier before starting the next inference round.
- Start treating each returned call as a stable result object. Persist its
  `tool_call_id`, `status`/`outcome`, `reason` or `error`, `request` lifecycle,
  `diagnostics`, `worker_id`, `admission`, `concurrency`, and `retry_count`.
- Start checking `toolbox.describe().parallel_execution` before choosing a
  concurrency limit. The effective runtime reports logical capacity, bounded
  queue depth and timeout, active/queued calls, worker process count, and the
  threaded-worker execution model.
- Start declaring tool policy when registering tools. The supported metadata is:

  ```json
  {
    "concurrency": {
      "mode": "parallel|serial|keyed|exclusive",
      "group": "optional-shared-group",
      "max_concurrency": 8,
      "key_argument": "resource_id"
    }
  }
  ```

  `serial` is one call at a time, `keyed` serializes calls with the same
  derived key while allowing different keys to overlap, and `exclusive` blocks
  other calls in its group. Auto-callable registration accepts the same
  `concurrency` object; manual tool definitions may place it on the function
  definition or tool definition.
- Start marking parallel Host API providers and methods as thread-safe. Use
  serial or keyed metadata for shared mutable clients, files, browser sessions,
  databases, and other protected resources. Host callbacks use bounded
  admission, queue timeout, cancellation, and provider/method identity fields.
- Start treating `queue_full`, `queue_timeout`, `host_call_queue_full`,
  `host_call_queue_timeout`, `host_call_canceled`, and `sandbox_recycled` as
  explicit outcomes in telemetry and retry policy.
- Keep the dependent project’s existing serialized operations serialized:
  scope/search mutations, approval decisions, protected-resource calls, and
  successive inference rounds remain barriers unless explicitly allowed by the
  declared policy.

## Compatibility defaults

Tools without concurrency metadata use the compatibility `parallel` policy.
Hosted toolbox registration defaults to a bounded queue with depth 32 and a
30-second queue wait; the effective logical capacity comes from the registration
capabilities (256 when unspecified). Host API methods default to bounded
admission with a method/provider group, depth 64, and a 30-second queue wait;
serial methods are capped at one in-flight call. These are runtime defaults,
not a promise that every underlying tool or provider is thread-safe.

No parent-chat batch policy change is required. The dependent project may keep
its current serialized batch application and opt into bounded concurrency only
for the calls it can safely overlap.
