# Features

This document is a capability-oriented map of what MP13 provides. It is split into:

- **Core engine** capabilities (production-oriented)

- **App/context layers** capabilities (testing/experimentation oriented today)


For architecture and data model orientation, see **[APPLAYERS.md](APPLAYERS.md)**.

---

## Core engine

### Inference (text-only)
- HF Transformers-based CausalLM inference
- Batch and single requests
- Optional streaming and per-item metrics
- Cache control per request and sensible engine defaults

### Async runtime, concurrency, cancellation
- Async-first execution model
- Cancellation support per request
- Concurrent scheduling of compatible requests
- Fair queueing between incompatible request groups
- Multi-engine instances in the same process with friendly aliases for side-by-side models or configs.

### Tools
- Tool calling with parsing and tool-block reporting
- Unified tool-call parsing across supported models
- Same semantics for testing and deployment

### Adapters (PEFT / LoRA)
- Mixed PEFT inference: multiple active adapters with per-prompt overrides
- LoRA-first training workflow
- Fast switching between training and inference without reloading the base model

### Performance controls (optional)
- Quantization options (stack-dependent)
- Optional Triton and FlashAttention 2 groups
- Multi-engine instances with friendly aliases

### Debugging and stability
- Prompt formatting visibility and token counting helpers
- Explicit debugging for prompts and adapter selection
- Metrics visibility and recent aggregated metrics

---

## App / context layers (MIT)

The app layers provide a full system implementation for testing and experimentation. It was not tested under stress or for all promised scenarios and may have bugs.

Capabilities:
- Session trees with branching (forks, batches, retries)
- Cursor-based navigation with scoped state resolution
- Session persistence (save/load) and replay
- CLI orchestration helpers and inspection commands
- Tool visibility/scoping and adapter stack resolution across branches

**API/design note:** app-layer abstractions may evolve based on community feedback.

---


## Training (Integrated PEFT)
- LoRA adapter training with text or chat datasets.
- Auto memory management and checkpointing support.
- Trainer precision: `bf16`, `fp16`, `fp32` (default: `bf16`).

## Inference
- Batch or single requests with optional streaming and per-item metrics.
- Async execution with request cancelation and control per item in a batch.
- Tool calling with parsing and tool-block reporting.
- Cache control per request and sensible engine defaults.

## API Capabilities
- Engine lifecycle, multi-instance routing, and default engine selection.
- Mode toggling between training and inference.
- Async inference execution, cancelation, and metrics visibility.
- Prompt formatting and token counting for debugging or tool-router (MCP-style) integration.
- Capability discovery for automated clients.

## Inference Request Parameters
**Input format**
- `raw_list`: **[not tested]** pre-formatted prompts (no default; mutually exclusive with `messages_list`).
- `messages_list`: chat messages (no default; mutually exclusive with `raw_list`).

**Adapters**
- `override_adapters`: per-prompt adapter list, `__base__` uses base model (default: none).
- `active_adapters`: activate adapters for this request; empty list forces base model (default: none).
- Per-request adapter settings help avoid semantic conflicts in concurrent requests.

**Generation and output**
- `generation_config`: generation overrides (default: none).
- `stream`: stream tokens (default: `false`).
- `return_prompt`: `full` | `last` (default: none).
- `suppress_full_response`: hide full response text (default: `false`).
- `do_continue`: continue from last assistant message (default: `false`).

**Tools and parsing**
- `tools`: tool definitions for tool calling (default: none).
- `no_tools_parse`: disable tool parsing for this request (default: `false`).

**Metrics and IDs**
- `request_id`: client-provided id (default: none).
- `reset_metrics`: clear inference metrics before running (default: `false`).

**Cache control**
- `cache`: `dynamic`, `static`, `offloaded`, `dynamic_reset`, `static_reset`, `no_cache` (default: none).

## Engine Initialization Config
**Required**
- `base_model_name_or_path`: local path or Hub id (no default).

**Identity and mode**
- `instance_id`: custom alias (default: none).
- `initial_engine_mode`: `inference` | `train` (default: `inference`).
- `custom_chat_template`: override tokenizer template (default: none).
- `tools_parser_profile_key`: select a tool parser profile (default: none).
- `no_tools_parse`: disable tool parsing globally (default: `false`).
- `disable_custom_pad_ids`: avoid changing tokenizer pad ids (default: `false`).

**Device and precision**
- `device_map`: `auto`, `cpu`, or a device map (default: `auto`).
- `base_model_torch_dtype`: `auto` | `float16` | `bfloat16` | `float32` (default: `auto`).
- `trust_remote_code`: allow model-provided code (default: `true`).

**Quantization**
- `quantize_bits`: `none` | `4` | `8` | `awq` | `hqq` | `eetq` (default: `none`).

**Context, caching, and performance**
- `attn_implementation`: `auto` | `sdpa` | `flash_attention_2` | `eager` (default: `auto`).
- `default_context_size`: derived from model when unset (default: `None`).
- `default_max_new_tokens`: max output tokens (default: `8192`).
- `use_cache`: enable KV cache (default: `true`).
- `static_kv_cache`: static cache (default: `false`).
- `use_torch_compile`: enable `torch.compile` (default: `true`).
- `concurrent_generate`: concurrent requests (default: `4`).

**Logging**
- `log_with_instance_id`: tag logs with instance id (default: `false`).
- `log_instance_id_width`: log column width (default: `8`).
