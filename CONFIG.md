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

Other helpers:
- `--print-raw`: print config file as-is
- `--print-set`: print only keys set in the config file
- `--force`: overwrite target config file when used with `--init` or `--interactive`
- `--reconfigure`: interactive edit of an existing config
- `--clone`: start from another config path/name
- `--merge-default`: merge current defaults into the target config

---

## Merge rules

- Custom config overrides default config.
- Missing keys fall back to default config.
- `category_dirs` in a custom config override only the keys they provide.

---

## Path resolution rules

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

## Reference (original rules)

Below is the original config rules reference as provided in this repo, preserved for completeness.

MP13 Config Rules

Default config
- Default config path: `~/.mp13-llm/mp13_config.json`
- Custom config: pass `--config` in chat/demo to merge on top of default

Config helper
- `python mp13config.py --interactive` runs a guided setup with explanations.
- `python mp13config.py --init` writes a template config.
- `python mp13config.py --print` prints the resolved config.
- `python mp13config.py --print-raw` prints the config file as-is.
- `python mp13config.py --print-set` prints only keys set in the config file.

Alternate:
- `python -m src.app.config --interactive`
- `--force` overwrites the target config when used with `--init` or `--interactive`.
- `--reconfigure` runs interactive edit against an existing config (implies `--interactive`).
- Use `--config` to set the target config path/name.
- `--clone` starts from another config path/name (use `default` for the default config).
- `--merge-default` merges current defaults into the target config.

Merge rules
- Custom config overrides default config.
- Missing keys fall back to default config.
- `category_dirs` in a custom config override only the keys they provide.

Path resolution rules
- Absolute paths stay absolute.
- Paths starting with `./` or `../` resolve relative to the current working directory.
- Other relative paths resolve relative to the category root.
- `~` expands to the user home directory.
- Anchors:
  - `@home` -> user home directory
  - `@temp` -> OS temp directory
  - `@project` -> git project root (or cwd if no git root)
  - `@config` -> directory of the config file
  - `@models`, `@adapters`, `@data`, `@tools`, `@sessions`, `@logs` -> category roots

Category roots
Set via `category_dirs` in config:
```
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

Model paths
- `base_model_path` is resolved as a local path (category-relative or anchored).

Tools config path
- `tools_config_path` lives under `engine_params` and is inherited by `inference_params`.
- If you need a conversation-specific override, set it explicitly inside `inference_params`.

Inference params
- `inference_params` is the per-conversation default parameter block for chat.
```
{
  "inference_params": {
    "stream": true,
    "cache": null,
    "return_prompt": null,
    "generation_config_template": {},
    "max_new_tokens": null,
    "no_tools_parse": false,
    "auto_retry_truncated": false,
    "suppress_full_response": false,
    "results_as_user_role": false,
    "pack_results_as_one_role": false,
    "advertised_tools": [],
    "silent_tools": [],
    "disabled_tools": [],
    "auto_tool_retry_limit": 5,
    "auto_continue_retry_limit": 10,
    "global_tools_mode": "advertised"
  }
}
```

Engine params
- Engine-level settings live under `engine_params` (base model path, quantization, tools config, log levels, engine defaults).
```
{
  "engine_params": {
    "base_model_path": "",
    "base_model_dtype": "auto",
    "quantize_bits": "none",
    "hqq_bits": 4,
    "hqq_group_size": 64,
    "hqq_quant_zero": true,
    "hqq_quant_scale": false,
    "hqq_axis": 1,
    "default_system_message": "",
    "default_context_size": null,
    "default_max_new_tokens": 8192,
    "attn_implementation": "auto",
    "use_cache": true,
    "device_map": "auto",
    "trust_remote_code": true,
    "use_torch_compile": true,
    "static_kv_cache": false,
    "concurrent_generate": 4,
    "tools_config_path": "mp13tools.json",
    "console_log_level": "warning",
    "file_log_level": "debug"
  }
}
```

Training params
- `training_params` provides default values for the training workflow when CLI flags are omitted.
- Supported keys: `training_steps`, `trainer_precision`, `lora_r`, `lora_alpha`, `lora_dropout`, `lora_target_modules`, `train_override_ctx`.

Examples
```
{
  "engine_params": {
    "base_model_path": "granite-3.3-2b-instruct",
    "tools_config_path": "mp13tools.json"
  },
  "category_dirs": {
    "models_root_dir": "D:/models",
    "tools_root_dir": "@project/configs"
  }
}
```
