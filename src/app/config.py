# Copyright (c) 2025 mp13
# Author: alexeiv-ai <188820640+alexeiv-ai@users.noreply.github.com>
# AI-Assistance: Portions of this file were drafted using AI coding tools
# (e.g., ChatGPT, Gemini, Codex) under active human design supervision.
# Contact: Please open an issue or discussion on GitHub.
# SPDX-License-Identifier: MIT
import argparse
import json
import importlib.util
from pathlib import Path
from typing import Optional, Tuple


def _load_config_paths_module():
    module_path = Path(__file__).resolve().parents[1] / "mp13_engine" / "mp13_config_paths.py"
    spec = importlib.util.spec_from_file_location("mp13_config_paths", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load mp13_config_paths from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_cfg = _load_config_paths_module()
DEFAULT_CATEGORY_DIRS = _cfg.DEFAULT_CATEGORY_DIRS
DEFAULT_CHAT_CONFIG = _cfg.DEFAULT_CHAT_CONFIG
get_default_config_dir = _cfg.get_default_config_dir
get_default_config_path = _cfg.get_default_config_path
load_json_config = _cfg.load_json_config
load_merged_config = _cfg.load_merged_config
resolve_config_paths = _cfg.resolve_config_paths
save_json_config = _cfg.save_json_config


def _resolve_target_path(args: argparse.Namespace) -> Path:
    if args.config:
        return _resolve_save_path(args.config, get_default_config_path(), add_suffix=True)
    return get_default_config_path()


def _resolve_save_path(raw: str, fallback: Path, *, add_suffix: bool = False) -> Path:
    if not raw:
        return fallback
    candidate = Path(raw)
    if candidate.is_absolute() or str(candidate).startswith(("./", ".\\", "../", "..\\")):
        resolved = candidate.expanduser().resolve()
    else:
        resolved = (get_default_config_dir() / candidate).expanduser().resolve()
    if add_suffix and resolved.suffix == "":
        resolved = resolved.with_suffix(".json")
    return resolved


def _build_template_config() -> dict:
    return json.loads(json.dumps(DEFAULT_CHAT_CONFIG))


def _ensure_category_dirs(config: dict) -> None:
    category_dirs = config.get("category_dirs")
    if not isinstance(category_dirs, dict):
        return
    for value in category_dirs.values():
        if not value:
            continue
        try:
            Path(value).mkdir(parents=True, exist_ok=True)
        except OSError:
            pass


def _print_intro() -> None:
    print("MP13 config interactive setup")
    print("- Empty input keeps the current value.")
    print("- Use ./ or ../ for cwd-relative paths.")
    print("- Use @home, @temp, @project, @config, @models, @adapters, @data, @tools, @sessions, @logs anchors.")
    print("- Empty fields in custom configs fall back to defaults.")
    print("- Avoid empty fields in defaults unless you want engine defaults.")


def _prompt_value(label: str, current: str, *, required: bool) -> str:
    while True:
        raw = input(f"{label} [{current}]: ").strip()
        if raw:
            return raw
        if current:
            return current
        if not required:
            return ""
        print("This value is required.")


def _prompt_value_with_default(label: str, current: str, default: str, *, is_set: bool, required: bool) -> str:
    if is_set:
        display = current
    else:
        display = f"<default:{default}>"
    while True:
        raw = input(f"{label} [{display}]: ").strip()
        if raw:
            return raw
        if is_set:
            return current
        if default:
            return default
        if not required:
            return ""
        print("This value is required.")


def _prompt_int(label: str, current: int, *, required: bool) -> int:
    while True:
        raw = input(f"{label} [{current}]: ").strip()
        if not raw:
            if current is not None:
                return int(current)
            if not required:
                return 0
        else:
            if raw.isdigit():
                return int(raw)
        print("Enter a valid integer value.")


def _parse_optional_int(value: str) -> Optional[int]:
    raw = value.strip()
    if not raw or raw.lower() == "none":
        return None
    if raw.isdigit():
        return int(raw)
    return None


def _strip_defaults(config: dict, defaults: dict) -> dict:
    if not isinstance(config, dict) or not isinstance(defaults, dict):
        return config
    result = {}
    for key, value in config.items():
        if key not in defaults:
            result[key] = value
            continue
        default_value = defaults.get(key)
        if isinstance(value, dict) and isinstance(default_value, dict):
            nested = _strip_defaults(value, default_value)
            if nested:
                result[key] = nested
        else:
            if value != default_value:
                result[key] = value
    return result


def _select_sections() -> dict:
    selections = {
        "category_dirs": False,
        "engine_params": False,
        "inference_params": False,
        "training_params": False,
        "log_settings": False,
    }
    sections = [
        ("category_dirs", "Category dirs"),
        ("engine_params", "Engine params"),
        ("inference_params", "Inference params"),
        ("training_params", "Training params"),
        ("log_settings", "Log settings"),
    ]
    print("Select sections to edit (comma list, 'all', or 'none'):")
    for idx, (_, label) in enumerate(sections, start=1):
        print(f"  {idx}) {label}")
    raw = input("Sections [all]: ").strip().lower()
    if not raw or raw == "all":
        for key in selections:
            selections[key] = True
        return selections
    if raw == "none":
        return selections
    chosen = {item.strip() for item in raw.split(",") if item.strip()}
    for idx, (key, _) in enumerate(sections, start=1):
        selections[key] = key in chosen or str(idx) in chosen
    return selections


def _interactive_config(
    source_path: Path,
    save_path: Path,
    *,
    prompt_for_path: bool,
    existing: dict,
    defaults: dict,
) -> Tuple[dict, Path]:
    category_dirs = dict(DEFAULT_CATEGORY_DIRS)
    if isinstance(existing.get("category_dirs"), dict):
        category_dirs.update(existing["category_dirs"])

    _print_intro()
    print(f"Source config path: {source_path}")
    if prompt_for_path:
        raw_path = input(f"Save config as (blank keeps {save_path}): ").strip()
        save_path = _resolve_save_path(raw_path, save_path)
    print(f"Save config path: {save_path}")

    selections = _select_sections()

    merged = dict(existing)

    if selections["category_dirs"]:
        defaults_dirs = defaults.get("category_dirs") or {}
        category_dirs["models_root_dir"] = _prompt_value_with_default(
            "Models root",
            category_dirs.get("models_root_dir", ""),
            defaults_dirs.get("models_root_dir", ""),
            is_set="category_dirs" in existing and "models_root_dir" in (existing.get("category_dirs") or {}),
            required=True,
        )
        category_dirs["adapters_root_dir"] = _prompt_value_with_default(
            "Adapters root",
            category_dirs.get("adapters_root_dir", ""),
            defaults_dirs.get("adapters_root_dir", ""),
            is_set="category_dirs" in existing and "adapters_root_dir" in (existing.get("category_dirs") or {}),
            required=True,
        )
        category_dirs["data_root_dir"] = _prompt_value_with_default(
            "Data root",
            category_dirs.get("data_root_dir", ""),
            defaults_dirs.get("data_root_dir", ""),
            is_set="category_dirs" in existing and "data_root_dir" in (existing.get("category_dirs") or {}),
            required=True,
        )
        category_dirs["sessions_root_dir"] = _prompt_value_with_default(
            "Sessions root",
            category_dirs.get("sessions_root_dir", ""),
            defaults_dirs.get("sessions_root_dir", ""),
            is_set="category_dirs" in existing and "sessions_root_dir" in (existing.get("category_dirs") or {}),
            required=True,
        )
        category_dirs["tools_root_dir"] = _prompt_value_with_default(
            "Tools root",
            category_dirs.get("tools_root_dir", ""),
            defaults_dirs.get("tools_root_dir", ""),
            is_set="category_dirs" in existing and "tools_root_dir" in (existing.get("category_dirs") or {}),
            required=True,
        )
        category_dirs["logs_root_dir"] = _prompt_value_with_default(
            "Logs root",
            category_dirs.get("logs_root_dir", ""),
            defaults_dirs.get("logs_root_dir", ""),
            is_set="category_dirs" in existing and "logs_root_dir" in (existing.get("category_dirs") or {}),
            required=True,
        )
        merged["category_dirs"] = category_dirs

    if selections["engine_params"]:
        engine_params = dict(merged.get("engine_params") or {})
        default_engine = defaults.get("engine_params") or {}
        engine_params["base_model_path"] = _prompt_value_with_default(
            "Base model path (optional, use hf:org/model for remote)",
            str(engine_params.get("base_model_path", "")),
            str(default_engine.get("base_model_path", "")),
            is_set="engine_params" in existing and "base_model_path" in (existing.get("engine_params") or {}),
            required=False,
        )
        engine_params["base_model_dtype"] = _prompt_value_with_default(
            "Base model dtype (auto, bfloat16, float16, float32)",
            str(engine_params.get("base_model_dtype", "auto")),
            str(default_engine.get("base_model_dtype", "auto")),
            is_set="engine_params" in existing and "base_model_dtype" in (existing.get("engine_params") or {}),
            required=True,
        )
        engine_params["default_system_message"] = _prompt_value_with_default(
            "Default system message (optional)",
            str(engine_params.get("default_system_message", "")),
            str(default_engine.get("default_system_message", "")),
            is_set="engine_params" in existing and "default_system_message" in (existing.get("engine_params") or {}),
            required=False,
        )
        default_ctx_raw = _prompt_value_with_default(
            "Default context size (optional)",
            str(engine_params.get("default_context_size", "")),
            str(default_engine.get("default_context_size", "")),
            is_set="engine_params" in existing and "default_context_size" in (existing.get("engine_params") or {}),
            required=False,
        )
        engine_params["default_context_size"] = _parse_optional_int(default_ctx_raw)
        engine_params["default_max_new_tokens"] = _prompt_int(
            "Default max new tokens",
            int(engine_params.get("default_max_new_tokens", 8192)),
            required=True,
        )
        engine_params["tools_config_path"] = _prompt_value_with_default(
            "Tools config path (relative uses tools root)",
            str(engine_params.get("tools_config_path", "mp13tools.json")),
            str(default_engine.get("tools_config_path", "mp13tools.json")),
            is_set="engine_params" in existing and "tools_config_path" in (existing.get("engine_params") or {}),
            required=True,
        )
        engine_params["console_log_level"] = _prompt_value_with_default(
            "Console log level (error, warning, info, debug, all, none)",
            str(engine_params.get("console_log_level", "warning")),
            str(default_engine.get("console_log_level", "warning")),
            is_set="engine_params" in existing and "console_log_level" in (existing.get("engine_params") or {}),
            required=True,
        )
        engine_params["file_log_level"] = _prompt_value_with_default(
            "File log level (error, warning, info, debug, all, none)",
            str(engine_params.get("file_log_level", "debug")),
            str(default_engine.get("file_log_level", "debug")),
            is_set="engine_params" in existing and "file_log_level" in (existing.get("engine_params") or {}),
            required=True,
        )
        merged["engine_params"] = engine_params

    if selections["inference_params"]:
        params = dict(merged.get("inference_params") or {})
        default_params = defaults.get("inference_params") or {}
        for key, value in (default_params or {}).items():
            params.setdefault(key, value)
        params["stream"] = _prompt_value_with_default(
            "Stream (True/False)",
            str(params.get("stream", True)),
            str(default_params.get("stream", True)),
            is_set="inference_params" in existing and "stream" in (existing.get("inference_params") or {}),
            required=True,
        ).lower() in ("true", "1", "yes", "y")
        params["max_new_tokens"] = _prompt_value_with_default(
            "Max new tokens (empty for None)",
            str(params.get("max_new_tokens", "")),
            str(default_params.get("max_new_tokens", "")),
            is_set="inference_params" in existing and "max_new_tokens" in (existing.get("inference_params") or {}),
            required=False,
        )
        if params["max_new_tokens"] == "" or params["max_new_tokens"].lower() == "none":
            params["max_new_tokens"] = None
        params["no_tools_parse"] = _prompt_value_with_default(
            "Disable tool parsing (True/False)",
            str(params.get("no_tools_parse", False)),
            str(default_params.get("no_tools_parse", False)),
            is_set="inference_params" in existing and "no_tools_parse" in (existing.get("inference_params") or {}),
            required=True,
        ).lower() in ("true", "1", "yes", "y")
        params["suppress_full_response"] = _prompt_value_with_default(
            "Suppress full response (True/False)",
            str(params.get("suppress_full_response", False)),
            str(default_params.get("suppress_full_response", False)),
            is_set="inference_params" in existing and "suppress_full_response" in (existing.get("inference_params") or {}),
            required=True,
        ).lower() in ("true", "1", "yes", "y")
        merged["inference_params"] = params

    if selections["training_params"]:
        tparams = dict(merged.get("training_params") or {})
        default_tparams = defaults.get("training_params") or {}
        for key, value in (default_tparams or {}).items():
            tparams.setdefault(key, value)
        tparams["training_steps"] = _prompt_int("Training steps", int(tparams.get("training_steps", 100)), required=True)
        tparams["trainer_precision"] = _prompt_value_with_default(
            "Trainer precision (bf16, fp16, fp32)",
            str(tparams.get("trainer_precision", "bf16")),
            str(default_tparams.get("trainer_precision", "bf16")),
            is_set="training_params" in existing and "trainer_precision" in (existing.get("training_params") or {}),
            required=True,
        )
        tparams["lora_r"] = _prompt_int("LoRA r", int(tparams.get("lora_r", 8)), required=True)
        tparams["lora_alpha"] = _prompt_int("LoRA alpha", int(tparams.get("lora_alpha", 64)), required=True)
        tparams["lora_dropout"] = float(_prompt_value_with_default(
            "LoRA dropout",
            str(tparams.get("lora_dropout", 0.0)),
            str(default_tparams.get("lora_dropout", 0.0)),
            is_set="training_params" in existing and "lora_dropout" in (existing.get("training_params") or {}),
            required=True,
        ))
        tparams["lora_target_modules"] = _prompt_value_with_default(
            "LoRA target modules (comma list or empty)",
            str(tparams.get("lora_target_modules") or ""),
            str(default_tparams.get("lora_target_modules") or ""),
            is_set="training_params" in existing and "lora_target_modules" in (existing.get("training_params") or {}),
            required=False,
        ) or None
        tparams["train_override_ctx"] = _prompt_value_with_default(
            "Train override ctx (auto or number)",
            str(tparams.get("train_override_ctx", "auto")),
            str(default_tparams.get("train_override_ctx", "auto")),
            is_set="training_params" in existing and "train_override_ctx" in (existing.get("training_params") or {}),
            required=True,
        )
        merged["training_params"] = tparams

    if selections["log_settings"]:
        category_dirs = dict(merged.get("category_dirs") or {})
        category_dirs["logs_root_dir"] = _prompt_value(
            "Logs root (empty for default)",
            str(category_dirs.get("logs_root_dir", "")),
            required=False,
        )
        merged["category_dirs"] = category_dirs

    resolved, _ = resolve_config_paths(merged, cwd=Path.cwd(), config_path=save_path)
    print("Resolved preview:")
    print(json.dumps(resolved, indent=2))
    confirm = input("Save this configuration? [y/N]: ").strip().lower()
    if confirm not in ("y", "yes"):
        print("Aborted.")
        return {}, save_path
    return merged, save_path


def main() -> int:
    parser = argparse.ArgumentParser(description="MP13 config helper")
    parser.add_argument("--config", type=str, default=None, help="Target config file path or name.")
    parser.add_argument("--clone", type=str, default=None, help="Clone from another config path or name ('default' uses the default config).")
    parser.add_argument("--init", action="store_true", help="Create a default config file at the target path.")
    parser.add_argument("--interactive", action="store_true", help="Run an interactive config wizard.")
    parser.add_argument("--reconfigure", action="store_true", help="Reconfigure an existing config file.")
    parser.add_argument("--merge-default", action="store_true", help="Merge current defaults into the target config.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing config when used with --init.")
    parser.add_argument("--print", dest="print_config", action="store_true", help="Print resolved config as JSON.")
    parser.add_argument("--print-raw", action="store_true", help="Print raw config file content without resolution.")
    parser.add_argument("--print-set", action="store_true", help="Print only keys set in the config file (no defaults).")
    args = parser.parse_args()

    target_path = _resolve_target_path(args)
    if args.clone:
        if args.clone.lower() == "default":
            source_path = get_default_config_path()
        else:
            source_path = _resolve_save_path(args.clone, get_default_config_path())
    else:
        source_path = target_path

    if args.clone and not (args.interactive or args.init or args.print_config or args.print_raw or args.print_set or args.merge_default):
        args.interactive = True

    if args.merge_default and not (args.print_config or args.print_raw or args.print_set or args.interactive or args.init):
        merged, _, _ = load_merged_config(
            default_config_path=get_default_config_path(),
            custom_config_path=source_path,
        )
        save_json_config(merged, target_path)
        print(f"Config merged with defaults at {target_path}")
        return 0

    if args.reconfigure:
        args.interactive = True

    if args.interactive:
        if target_path.exists() and not args.force and not args.reconfigure:
            print(f"Config already exists at {target_path}. Use --force to overwrite.")
            return 1
        defaults = _build_template_config()
        existing = load_json_config(source_path) or {}
        if args.clone and not args.merge_default:
            existing = _strip_defaults(existing, defaults)
        elif args.merge_default:
            existing = load_merged_config(
                default_config_path=get_default_config_path(),
                custom_config_path=source_path,
            )[0]
        wizard_config, final_path = _interactive_config(
            source_path,
            target_path,
            prompt_for_path=False,
            existing=existing,
            defaults=defaults,
        )
        if wizard_config:
            save_json_config(wizard_config, final_path)
            resolved, _ = resolve_config_paths(wizard_config, cwd=Path.cwd(), config_path=final_path)
            _ensure_category_dirs(resolved)
            print(f"Config written to {final_path}")
        return 0

    if args.init:
        if target_path.exists() and not args.force:
            print(f"Config already exists at {target_path}. Use --force to overwrite.")
            return 1
        defaults = _build_template_config()
        if args.clone:
            source_cfg = load_json_config(source_path) or {}
            if not args.merge_default:
                source_cfg = _strip_defaults(source_cfg, defaults)
            else:
                source_cfg = load_merged_config(
                    default_config_path=get_default_config_path(),
                    custom_config_path=source_path,
                )[0]
            save_json_config(source_cfg, target_path)
            resolved, _ = resolve_config_paths(source_cfg, cwd=Path.cwd(), config_path=target_path)
            _ensure_category_dirs(resolved)
            print(f"Config written to {target_path}")
            return 0
        template = defaults
        save_json_config(template, target_path)
        resolved, _ = resolve_config_paths(template, cwd=Path.cwd(), config_path=target_path)
        _ensure_category_dirs(resolved)
        print(f"Config written to {target_path}")
        return 0


    if args.print_raw:
        raw = load_json_config(source_path) if args.clone_from else (load_json_config(target_path) or {})
        raw = raw or {}
        print(json.dumps(raw, indent=2))
        return 0

    if args.print_set:
        raw = load_json_config(source_path) if args.clone_from else (load_json_config(target_path) or {})
        raw = raw or {}
        print(json.dumps(raw, indent=2))
        return 0

    if args.print_config:
        if args.merge_default and target_path != get_default_config_path():
            merged, _, _ = load_merged_config(
                default_config_path=get_default_config_path(),
                custom_config_path=source_path,
            )
            config = merged
        else:
            config = load_json_config(source_path) or {}
        resolved, _ = resolve_config_paths(
            config,
            cwd=Path.cwd(),
            config_path=target_path,
        )
        print(json.dumps(resolved, indent=2))

    if not args.init and not args.print_config:
        print(f"Target config path: {target_path}")
        print("Use --init to create or --print to resolve.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
