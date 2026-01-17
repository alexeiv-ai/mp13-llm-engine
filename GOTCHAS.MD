# Gotchas

Known caveats and behavior notes:

* **Replay scope:** Only a subset of chat commands is replayed into the destination conversation. Most API-style commands and all log commands are ignored.
* **Replay permissions:** Global toolbox permission commands (for example, `/tools a`) are not replayed for safety when sharing sessions. In a source session, enable only needed tools in the global toolbox up front. In a destination session, allow only approved tools via config or manually before replaying a foreign session.
* **Replay flags:** Global flag commands are not replayed to allow source and destination configs to differ (for example, streaming or auto-truncation in the source can be suppressed in the destination).
* **Replay semantics:** Replay mainly supports commands that affect dynamic scoped resolution such as system messages, active adapters, and scoped tool permissions (subject to what the global toolbox exposes).
* **Replay comparisons:** A few API commands are supported to compare perf/memory metrics and raw prompts between source and destination. Metrics comparisons can be done with `/s hfs` run on the source and the replayed conversations.
* **Tools parsing on truncation:** When the engine detects a truncated response, the app layer bails out so tool calls are not auto-executed even after a client requests a continuation. A manual app-layer call into the tools parser is required (not yet implemented). Note that response chunks always have tool blocks cut off the chunks text while complete response response text sent at the end have those blocks as part of its content.
* **Static Cache limits:** Static Cache does not deliver full background warm-up or prevent inference requests from blocking. Use it and only for stable prompt shapes and after manual warm-up; speedups can reach about 50%. Concurrency has to be set to 1 explicitly.
* **Static Cache and FlashAttention:** `flash_attn_2` can conflict with Static Cache.
* **Adapter activation:** The engine supports global and per-request adapter activation. Both work, but concurrency may break semantics with global activation. Prefer per-request activation (no perf loss); use global adapters only if they will not change during a session or you know what you are doing.
* **Batch streaming:** HF does not support response streaming for batch requests. If the engine receives a streaming argument for a batch, it splits into the first prompt (streamed) and the rest of the batch (non-streamed).
* **Multi-GPU behavior:** Multiple GPUs are supported, but this can negatively impact inference performance. Some models also do not behave well when sharded across GPUs. Unless you are constrained by VRAM, prefer loading the model on a single selected GPU.
* **Quantization tradeoffs:** Quantizing on load (where supported) saves VRAM but usually hurts speed. Linux-only quantization schemes have not been tested yet but are planned before v1.0. The current plan is to support only Mixed PEFT-friendly schemes, which are limited today.
* **Raw prompts:** The "raw prompts" engine feature has not been tested at all.
* **Special tokens:** The engine trims only EOS and PAD tokens, leaving all other special tokens to the app's discretion.
* **Generate args:** The engine does not expose non-standard args to HF `generate`, like `thinking=true`. Only what `GenerationConfig` supports is passed through.
* **PAD token behavior:** By default, the engine reserves a per-model stable PAD token if the model config sets PAD to EOS. This can be disabled, but there is evidence that removing PAD leads to LoRA adapter stability issues. I would rather not modify model PAD tokens without clear counter-arguments.
* **WSL model paths:** When using WSL, keep models on the native Linux partition (not mounted) or model load times will suffer beyond reasonable.
