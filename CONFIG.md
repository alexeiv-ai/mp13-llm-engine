# Config

MP13 uses a single JSON config to define model paths, runtime defaults, and directory layout. The same config model is intended to support both **local testing** and **production deployments** of the core engine.

Default config path:
- `~/.mp13-llm/mp13_config.json`

Custom config path:
- Pass `--config` in chat/demo to merge on top of default

---

## Quick start

Interactive setup (recommended):
```bash
python mp13config.py --interactive
```

Generate a template:
```bash
python mp13config.py --init
```

Inspect resolved config:
```bash
python mp13config.py --print
```

Examples:
```bash
# Create a custom config (empty unless --clone is provided)
python mp13config.py --init --config my_setup

# Clone defaults into a new config
python mp13config.py --init --config my_setup --clone default

# Merge missing values from another config
python mp13config.py --merge base_config --config my_setup

# Show overrides relative to default
python mp13config.py --diff default --config my_setup

# Update specific fields
python mp13config.py --reconfigure engine_params.base_model_path=granite-3.3-2b-instruct \
  inference_params.stream=true --config my_setup
```

Other helpers:
- `--print-raw`: print config file as-is
- `--print-set`: print only keys set in the config file
- `--force`: overwrite target config file when used with `--init` or `--interactive`
- `--reconfigure key=value ...`: update config values by key path (dot notation)
- `--clone`: start from another config path/name
- `--merge <base_config>`: merge missing values from a base config into the target
- `--diff <base_config>`: show config differences (use `default` to show overrides)

---

## Merge rules

- Custom config overrides default config.
- Missing keys fall back to default config.
- `category_dirs` in a custom config override only the keys they provide.

---

## Path resolution rules

- Config names (used with `--config`, `--clone`, `--merge`, `--diff`) default to `.json` and resolve relative to the current working directory.
- Absolute paths stay absolute.
- Paths starting with `./` or `../` resolve relative to the current working directory.
- Other relative paths resolve relative to the category root.
- `~` expands to the user home directory.

Anchors:
- `@home` -> user home directory
- `@temp` -> OS temp directory
- `@project` -> git project root (or cwd if no git root)
- `@config` -> directory of the config file
- `@models`, `@adapters`, `@data`, `@tools`, `@sessions`, `@logs` -> category roots

---

## Category roots

Set via `category_dirs` in config, for example:

```json
{
  "category_dirs": {
    "models_root_dir": "@project/..",
    "adapters_root_dir": "@project/adapters",
    "data_root_dir": "@project/data",
    "sessions_root_dir": "@home/.mp13-llm/sessions",
    "tools_root_dir": "@project/configs",
    "logs_root_dir": "@temp"
  }
}
```

---

## Engine params

Engine-level settings live under `engine_params` (base model path, quantization, tools config, logging).

Key fields (representative):
- `base_model_path`
- `quantize_bits`
- `attn_implementation`
- `device_map`
- `use_cache`
- `use_torch_compile`
- `concurrent_generate`
- `tools_config_path`
- log levels

---

## Inference params

`inference_params` defines per-conversation/request defaults for chat and app layers. These defaults combine with cursor overrides and per-turn changes.

Representative keys:
- `stream`
- `cache`
- `return_prompt`
- `max_new_tokens`
- `no_tools_parse`
- `auto_retry_truncated`
- tools mode + allow/deny lists

---

## Training params

`training_params` supplies default values for the training workflow when CLI flags are omitted.

Supported keys include:
- `training_steps`
- `trainer_precision` (`bf16`, `fp16`, `fp32`)
- `lora_r`, `lora_alpha`, `lora_dropout`
- `lora_target_modules`
- `train_override_ctx`

---

## Notes on production use

The core engine is intended to be reused between testing and deployment, so you can validate:

- model path resolution

- tool parsing behavior

- adapter storage conventions

- concurrency/cancellation behavior


…in local runs before packaging the same setup into a service.

---

## Appendix: Generating A Fresh Reference

The config surface evolves, so instead of a static reference block, generate a fresh copy from the CLI:

```bash
# Default config template (all supported keys)
python mp13config.py --init

# Resolved view of the default config (paths expanded)
python mp13config.py --print

# Resolved view of a custom config (merged with defaults)
python mp13config.py --print --config my_setup
```

Use `--print-set` on your config to see only explicitly set keys.
- Examples:
```
python mp13config.py --init --config my_setup
python mp13config.py --init --config my_setup --clone default
python mp13config.py --merge base_config --config my_setup
python mp13config.py --diff default --config my_setup
python mp13config.py --reconfigure engine_params.base_model_path=granite-3.3-2b-instruct inference_params.stream=true --config my_setup
```
