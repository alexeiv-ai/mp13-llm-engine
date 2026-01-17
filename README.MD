# MP13-LLM-Engine

MP13-LLM-Engine is a **local LLM engine** for **text-only inference** and integrated **LoRA (PEFT) training**, built on top of Hugging Face Transformers. It features **unified tool-call parsing** and is designed for people who want a solid, reusable engine for both standard and **agentic workflows** with per-request switching of multiple active LoRA adapters as supported by **Mixed PEFT** package.


> **Status:** v0.9 (public validation). A 1.0 release and Python wheel are planned.

---

## What MP13 is (and isn't)

MP13 deliberately separates two concerns:

* a **core engine** that focuses on mixed PEFT adoption, text generation transparency, switching between tuning inference and ease of integration
* **app/context layers** that make experimentation, testing, debugging, and branching conversational workflows easier

The core engine is production-oriented. The app layers are currently more research- and experimentation-focused.

---

## Core engine (`mp13_engine`)

The core engine is **not a CLI**. It is meant to be embedded into services, scripts, or other applications.

Design goals:

*   **Adoption & integration first:** Provide defaults that "just work" with minimal glue code.
* Stay close to **native Hugging Face Transformers** APIs with minimal extra dependencies treating all supported models as equal whenever possible 
* **Integrate and unify** tool calls parsing with configurable generalized profiles.
* Support for **performance optimization** features such as concurrency, batching, torch compile, tuned attention, kv cache while keeping all of them optional, not mandatory
*   **LLM Response stability & debuggability:** Focus on prompts encoding consistency, adapters visibility and LLM behavior tracking when things go wrong (prompts, adapters, tools).
* Use the same engine codepaths for **testing and deployment**


Runtime and performance characteristics:

* Async execution down to the HF transformers `generate` call
* Concurrent execution of compatible requests
* Fair queueing between incompatible request groups (for example, different adapter sets or training vs inference)
* Async cancellation as a first-class feature
* Per-request metrics plus rolling aggregated metrics from recent requests

Tool-call parsing is integrated into the engine so that behavior is consistent between local testing and production deployments. Tool registration and auto-execution are unified but have no dependencies on the engine itself.

---

## App / context layers (MIT-licensed)

The app layers have minimal to no dependencies on the engine and introduce a potential paradigm for dynamic **context-engineering**. These API layers are used heavily for engine testing but also carry promises beyond just that (though not completely tested due to vast test matrix).

Today, they are mainly used for:

* engine testing and validation
* research and experimentation
* exploring agentic and dynamic conversation branching workflows

They provide:

* session trees with branching, tryouts, retries, and replay
* automatic tools execution
* cursor-based navigation and dynamic turn-scoped conversation state resolution
* a CLI for interactive exploration and debugging

**API/design note:** the app-layer APIs may change as the design evolves or be replaced with something drastically different as part of my next project.

More detail:

* Architecture and internals: **[APPLAYERS.md](APPLAYERS.md)**
* Feature overview: **[FEATURES.md](FEATURES.md)**
* Gotchas and caveats: **[GOTCHAS.MD](GOTCHAS.MD)**

---

## Scope and limitations

MP13 intentionally limits scope to keep the engine focused and predictable:

* Hugging Face Transformers runtime
* CausalLM models
* **Text-only** generation
* safetensors format (no GGUF)
* quantization kinds are limited to **Mixed PEFT** compatible only

---

## Tested model families

Known to work with any of the below models, subject to compatibility with settings like attention, multi-GPU, and quantization:

* Qwen2.5-7B-Instruct-1M, Qwen3-8B
* DeepSeek-R1-Distill-Qwen-1.5B, DeepSeek-R1-Distill-Qwen-14B
* Granite 3.2 / 3.3 / Granite-3B-Code
* Mistral-Nemo-Instruct-2407, Ministral-3-3B-Instruct-2512
* Gemma-3-4B-IT
* Phi-3-Small-128K-Instruct, Phi-3.5-Mini-Instruct, Phi-4-Mini-Instruct

> Tool support depends on the underlying model. Reports of what works (or doesn't) in specific configurations are welcome.

---

## Quick start

### 1) Install

This repository uses **Poetry** and supports two pinned dependency stacks (mainline and legacy).

**Prerequisites (recommended):**
- Python 3.12+
- Poetry installed
- For GPU support: A compatible NVIDIA driver and CUDA runtime for your PyTorch build.

Minimal install:

```bash
poetry sync
```

Activate the virtual environment before running any commands:

```bash
# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate
```

Optional extras (platform-dependent):

```bash
poetry sync --with triton
poetry sync --with flashattn
poetry sync --with mistral3
```

> `--with mistral3` is only required for **Ministral-3-3B-Instruct-2512**.

* Full installation notes: **[INSTALL.md](INSTALL.md)**

---

### 2) Configure

Create or edit a config file:

```bash
python mp13config.py (use --interactive to go through each setting and provide a custom value)
```

Reminder: activate your virtual environment first.

Alternate:

```bash
python -m src.app.config (use --interactive to go through each setting and provide a custom value)
```

Default 'root' config path:

```
~/.mp13-llm/mp13_config.json
```

Derived configs are supported by resolving missing settings through the root config, see details:
* Config details: **[CONFIG.md](CONFIG.md)**

---

### 3) Run the engine demo

The demo trains a small test LoRA adapter and switches back to inference using the same engine setup.

```bash
py demo/demo_mp13_engine.py --base-model <model_name_or_path>
```

By default, it uses the 'root' config.
Run with `--help` to see all options. The trained adapter is saved under the configured adapters directory and should become visible through core engine API as well as in CLI chat when used with the same or compatible model. Note that different model families or quantized schemes may train an adapter using the same name, the engine will resolve only compatible adapters when loading by their names.

---

### 4) Use the chat app

```bash
python mp13chat.py
```

Reminder: activate your virtual environment first.

Alternate:

```bash
python -m src.app.mp13chat
```

By default, it uses the 'root' config.
Run with `--help` to see all options.
Inside chat, run `/?` to list supported commands, some commands support further '?' for scoped details.

If the workflow feels unfamiliar, it's often effective to point a capable LLM at the chat or app-layer source code and ask:

* What does this command do and why do I see this output?
* How would I implement workflow X?
* How can I host this app to register my own tools?
* (speculating here) Can I build a graphical UI that will make it easier managing, switching and tracking branched conversations using cursors and session tree related iterators?


For debugging or research, feeding same LLM extra context such as chat console output, a session tree, or a stack trace is usually enough to make progress.

---

## Versioning and expectations

### Before v1.0

Planned work:

* Fix engine bugs that block real deployments
* Address only the most problematic app-layer design issues, largely driven by community feedback
* Add a small set of tests built around the replay system

### After v1.0
Any post-V1 work under this project is subject to future time and resources constraints.
- Nice to have features:
  - More LoRAs, subject to a fuller test matrix
  - "Edge" device support (for example, MacBook or Qualcomm)
  - A polished training codepath and example adapters (will consider community provided datasets)
  - A graphical UI alternative to the CLI (may adopt from the community)

A separate **next project**, built on top of **MP13-engine**, will introduce new major features like LLM personalization, integrated training pipeline, passing documents in prompt, RAG and similar capabilities. Parts of those may later be back-ported to this project when making sense.

---

## Codebase note

Large portions of the project including .md files were written with help from LLMs (ChatGPT app, VsCode plugins Cline, Continue, Gemini Assist), with the final "last mile" mostly handled by **OpenAI Codex**. Some parts of the codebase may be difficult for developers to read but hopefully, for many cases, it may not be necessary.

The upside is that LLMs tend to be very effective at:

* explaining this project architecture
* investigating bugs or unexpected behavior
* proposing and implementing changes

This is intentional, as many power users and applied researchers may wear a developer hat for adopting the project to their specific needs.

---

## Community & Support

- 🐞 Bugs & concrete issues: https://github.com/alexeiv-ai/mp13-llm-engine/issues
- 💡 Ideas, questions, and design discussions: https://github.com/alexeiv-ai/mp13-llm-engine/discussions


## License

* Core engine (`mp13_engine/`): **Apache-2.0**
* App/runtime layers: **MIT**

Source files include SPDX identifiers; this README serves as the high-level license map.
