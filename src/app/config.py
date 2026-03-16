# Copyright (c) 2025 mp13
# Author: alexeiv-ai <188820640+alexeiv-ai@users.noreply.github.com>
# AI-Assistance: Portions of this file were drafted using AI coding tools
# (e.g., ChatGPT, Gemini, Codex) under active human design supervision.
# Contact: Please open an issue or discussion on GitHub.
# SPDX-License-Identifier: MIT
import argparse
import ast
import json
import importlib.util
import os
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


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
get_default_config_path = _cfg.get_default_config_path
resolve_config_path = _cfg.resolve_config_path
load_json_config = _cfg.load_json_config
deep_merge_dicts = _cfg.deep_merge_dicts
resolve_config_paths = _cfg.resolve_config_paths
save_json_config = _cfg.save_json_config
strip_defaults = _cfg.strip_defaults
diff_configs = _cfg.diff_configs
detect_project_root = _cfg.detect_project_root
PathResolver = _cfg.PathResolver
get_nested_value = _cfg.get_nested_value
set_nested_value = _cfg.set_nested_value
delete_nested_value = _cfg.delete_nested_value
get_hosting_control_state_path = _cfg.get_hosting_control_state_path
normalize_hosting_config_selector = _cfg.normalize_hosting_config_selector


def _resolve_target_path(args: argparse.Namespace, *, cwd: Path) -> Path:
    return resolve_config_path(args.config, cwd=cwd, default_config_path=get_default_config_path())


def _parse_csv_list(raw: Optional[str]) -> List[str]:
    if not str(raw or "").strip():
        return []
    return [x.strip() for x in str(raw).split(",") if x.strip()]


def _resolve_host_secret(args: argparse.Namespace) -> str:
    direct = str(getattr(args, "host_auth_secret", "") or "").strip()
    if direct:
        return direct
    env_name = str(getattr(args, "host_auth_secret_env", "") or "").strip()
    if env_name:
        return str(os.environ.get(env_name) or "").strip()
    if bool(getattr(args, "host_auth_secret_stdin", False)):
        return str(input()).strip()
    return ""


def _run_host_auth_ops(args: argparse.Namespace) -> Optional[int]:
    host_action = bool(
        args.host_auth_status
        or args.host_auth_list_keys
        or bool(args.host_auth_generate_secret)
        or bool(args.host_auth_upsert_key)
        or bool(args.host_auth_revoke_key)
        or bool(args.host_auth_issue_session)
        or bool(args.host_auth_revoke_session)
        or (args.host_auth_require_auth is not None)
    )
    if not host_action:
        return None

    from hosting.engine_host_service import EngineHostService

    control_state = Path(args.host_control_state_file).expanduser().resolve() if args.host_control_state_file else get_hosting_control_state_path().expanduser().resolve()
    svc = EngineHostService(control_state_file=control_state)

    if args.host_auth_generate_secret:
        print(json.dumps({"secret": secrets.token_urlsafe(int(args.host_auth_generate_secret))}, indent=2))
        return 0

    if args.host_auth_status:
        print(json.dumps(svc.auth_status(), indent=2))
        return 0

    if args.host_auth_list_keys:
        print(json.dumps(svc.auth_list_keys(), indent=2))
        return 0

    if args.host_auth_revoke_key:
        print(json.dumps(svc.auth_revoke_key(str(args.host_auth_revoke_key)), indent=2))
        return 0

    if args.host_auth_revoke_session:
        print(json.dumps(svc.auth_revoke_session(str(args.host_auth_revoke_session)), indent=2))
        return 0

    if args.host_auth_require_auth is not None:
        cfg = svc.set_control_config(require_auth=bool(args.host_auth_require_auth))
        print(json.dumps(cfg, indent=2))
        return 0

    if args.host_auth_upsert_key:
        key_id = str(args.host_auth_upsert_key).strip()
        role = str(args.host_auth_role or "").strip().lower()
        valid_roles = {
            "admin",
            "config_editor",
            "worker_user",
            "model_user_with_model_control",
            "model_user",
            "diagnostic_user",
        }
        if role not in valid_roles:
            print(
                "host-auth-upsert-key requires --host-auth-role "
                "{admin|config_editor|worker_user|model_user_with_model_control|model_user|diagnostic_user}"
            )
            return 1
        secret = _resolve_host_secret(args)
        if not secret:
            print("Missing key secret. Use --host-auth-secret, --host-auth-secret-env, or --host-auth-secret-stdin.")
            return 1
        allowed_configs = []
        if role == "config_editor":
            try:
                allowed_configs = [normalize_hosting_config_selector(x) if x != "*" else "*" for x in _parse_csv_list(args.host_auth_allowed_configs)]
            except ValueError as exc:
                print(f"Invalid --host-auth-allowed-configs: {exc}")
                return 1
        allowed_engines = []
        if role in {"worker_user", "model_user_with_model_control", "model_user"}:
            allowed_engines = _parse_csv_list(args.host_auth_allowed_engines)
        out = svc.auth_upsert_key(
            key_id=key_id,
            key_secret=secret,
            role=role,
            allowed_configs=allowed_configs,
            allowed_engines=allowed_engines,
            disabled=bool(args.host_auth_disable_key),
        )
        print(json.dumps(out, indent=2))
        return 0

    if args.host_auth_issue_session:
        key_id = str(args.host_auth_issue_session).strip()
        secret = _resolve_host_secret(args)
        if not secret:
            print("Missing key secret. Use --host-auth-secret, --host-auth-secret-env, or --host-auth-secret-stdin.")
            return 1
        scope = str(args.host_auth_scope or "control").strip().lower()
        if scope not in {"control", "config", "traffic"}:
            print("Invalid --host-auth-scope. Use one of: control, config, traffic.")
            return 1
        config_paths: List[str] = []
        if scope == "config":
            try:
                config_paths = [normalize_hosting_config_selector(x) if x != "*" else "*" for x in _parse_csv_list(args.host_auth_config_paths)]
            except ValueError as exc:
                print(f"Invalid --host-auth-config-paths: {exc}")
                return 1
        engine_ids = _parse_csv_list(args.host_auth_engine_ids) if scope == "traffic" else []
        out = svc.auth_issue_session(
            key_id=key_id,
            key_secret=secret,
            scope=scope,
            ttl_seconds=int(args.host_auth_ttl_seconds or 900),
            config_paths=config_paths,
            engine_ids=engine_ids,
        )
        print(json.dumps(out, indent=2))
        return 0

    print("No host auth action provided.")
    return 1


def _build_template_config() -> dict:
    return json.loads(json.dumps(DEFAULT_CHAT_CONFIG))


def _load_config_or_default(path: Path, *, default_path: Path, defaults: dict) -> dict:
    data = load_json_config(path)
    if data is None and path == default_path:
        return json.loads(json.dumps(defaults))
    return data or {}


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
    print("- Use numbered menus to select sections and fields.")
    print("- Special inputs: #=default/reset, !=save+quit, ^=quit.")
    print("- None/null values mean engine auto behavior (reset to global also uses that).")
    print("- Use ./ or ../ for cwd-relative paths.")
    print("- Use @home, @temp, @project, @config, @models, @adapters, @data, @tools, @sessions, @logs anchors.")
    print('- Use "" to set an empty string.')
    print("- Use # to reset to the global config (or engine default if no global value).")
    print("- For long text fields, use @<path> to load file contents (anchors supported).")


def _parse_optional_int(value: str) -> Optional[int]:
    raw = value.strip()
    if not raw or raw.lower() in {"none", "null"}:
        return None
    if raw.isdigit():
        return int(raw)
    return None


def _has_nested(config: Dict[str, Any], path: Tuple[str, ...]) -> bool:
    current: Any = config
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return False
        current = current[key]
    return True



@dataclass(frozen=True)
class FieldSpec:
    path: Tuple[str, ...]
    label: str
    description: str
    kind: str
    allowed: Optional[List[str]] = None
    allow_unknown: bool = False


SECTION_SPECS: List[Tuple[str, str, List[FieldSpec]]] = [
    (
        "category_dirs",
        "Category dirs",
        [
            FieldSpec(("category_dirs", "models_root_dir"), "Models root", "Root folder for base models.", "text"),
            FieldSpec(("category_dirs", "adapters_root_dir"), "Adapters root", "Root folder for adapters.", "text"),
            FieldSpec(("category_dirs", "data_root_dir"), "Data root", "Root folder for datasets.", "text"),
            FieldSpec(("category_dirs", "sessions_root_dir"), "Sessions root", "Root folder for session outputs.", "text"),
            FieldSpec(("category_dirs", "tools_root_dir"), "Tools root", "Root folder for tools config.", "text"),
            FieldSpec(("category_dirs", "logs_root_dir"), "Logs root", "Root folder for log files.", "text"),
        ],
    ),
    (
        "engine_params",
        "Engine params",
        [
            FieldSpec(("engine_params", "instance_id"), "Instance id", "Optional engine instance identifier.", "text"),
            FieldSpec(("engine_params", "base_model_path"), "Base model path", "Model folder name under models root or absolute path. Examples: llama-3.1-8b, D:/models/llama-3.1-8b.", "text"),
            FieldSpec(("engine_params", "base_model_dtype"), "Base model dtype", "Torch dtype for the base model.", "enum", ["auto", "bfloat16", "float16", "float32"]),
            FieldSpec(
                ("engine_params", "quantize_bits"),
                "Quantize bits",
                "Quantization mode for the base model.",
                "enum",
                ["none", "hqq", "eetq", "te"],
                allow_unknown=True,
            ),
            FieldSpec(("engine_params", "te_fp8_inference"), "TE FP8 inference", "Enable TE FP8 autocast in inference mode when quantize_bits='te'.", "bool"),
            FieldSpec(("engine_params", "te_fp8_training"), "TE FP8 training", "Enable TE FP8 autocast in training mode when quantize_bits='te'.", "bool"),
            FieldSpec(("engine_params", "hqq_bits"), "HQQ bits", "HQQ quantization bits.", "int"),
            FieldSpec(("engine_params", "hqq_group_size"), "HQQ group size", "HQQ group size.", "int"),
            FieldSpec(("engine_params", "hqq_quant_zero"), "HQQ quant zero", "Enable HQQ zero-point.", "bool"),
            FieldSpec(("engine_params", "hqq_quant_scale"), "HQQ quant scale", "Enable HQQ quant scale.", "bool"),
            FieldSpec(("engine_params", "hqq_axis"), "HQQ axis", "HQQ axis (0 or 1).", "int"),
            FieldSpec(("engine_params", "tools_parser_profile_key"), "Tools parser profile", "Tool parser profile key.", "text"),
            FieldSpec(("engine_params", "custom_chat_template"), "Custom chat template", "Override tokenizer chat template. Supports @file input.", "text"),
            FieldSpec(("engine_params", "initial_engine_mode"), "Initial engine mode", "Initial engine mode after init.", "enum", ["train", "inference"]),
            FieldSpec(("engine_params", "memory_mode"), "Memory mode", "Base model placement strategy.", "enum", ["auto_cpu", "single_gpu", "respect_device_map"]),
            FieldSpec(("engine_params", "default_system_message"), "Default system message", "Default system prompt prefix. Supports @file input.", "text"),
            FieldSpec(("engine_params", "default_context_size"), "Default context size", "Max context size for the engine.", "optional_int"),
            FieldSpec(("engine_params", "default_max_new_tokens"), "Default max new tokens", "Default max_new_tokens for inference.", "int"),
            FieldSpec(("engine_params", "attn_implementation"), "Attention implementation", "Attention backend.", "enum", ["auto", "sdpa", "flash_attention_2", "eager"]),
            FieldSpec(("engine_params", "use_cache"), "Use cache", "Enable KV cache.", "bool"),
            FieldSpec(("engine_params", "device_map"), "Device map", "Device placement strategy. Examples: auto, cpu, cuda:0, {\"\": 0}.", "json_or_text"),
            FieldSpec(("engine_params", "trust_remote_code"), "Trust remote code", "Allow remote model code.", "bool"),
            FieldSpec(("engine_params", "use_torch_compile"), "Use torch.compile", "Enable torch.compile for inference.", "bool"),
            FieldSpec(("engine_params", "static_kv_cache"), "Static KV cache", "Enable static KV cache.", "bool"),
            FieldSpec(("engine_params", "use_separate_stream"), "Separate CUDA stream", "Use a dedicated CUDA stream.", "bool"),
            FieldSpec(("engine_params", "concurrent_generate"), "Concurrent generate", "Max concurrent requests.", "int"),
            FieldSpec(("engine_params", "disable_custom_pad_ids"), "Disable custom pad ids", "Suppress pad token adjustments.", "bool"),
            FieldSpec(("engine_params", "no_tools_parse"), "No tools parse", "Disable tool parsing globally.", "bool"),
            FieldSpec(("engine_params", "tools_config_path"), "Tools config path", "Tools config file path.", "text"),
            FieldSpec(("engine_params", "console_log_level"), "Console log level", "Console log verbosity.", "enum", ["error", "warning", "info", "debug", "all", "none"]),
            FieldSpec(("engine_params", "file_log_level"), "File log level", "File log verbosity.", "enum", ["error", "warning", "info", "debug", "all", "none"]),
            FieldSpec(("engine_params", "log_with_instance_id"), "Log with instance id", "Include instance id in logs.", "bool"),
            FieldSpec(("engine_params", "log_instance_id_width"), "Log instance id width", "Fixed width for instance id column.", "int"),
        ],
    ),
    (
        "inference_params",
        "Inference params",
        [
            FieldSpec(("inference_params", "stream"), "Stream", "Stream tokens as they are generated.", "bool"),
            FieldSpec(("inference_params", "cache"), "Cache mode", "Cache mode override.", "text"),
            FieldSpec(("inference_params", "return_prompt"), "Return prompt", "Return prompt content.", "enum", ["full", "last"]),
            FieldSpec(("inference_params", "generation_config_template"), "Generation config", "Base generation config object. Supports @file input.", "json"),
            FieldSpec(("inference_params", "max_new_tokens"), "Max new tokens", "Override max_new_tokens.", "optional_int"),
            FieldSpec(("inference_params", "no_tools_parse"), "No tools parse", "Disable tool parsing.", "bool"),
            FieldSpec(("inference_params", "auto_retry_truncated"), "Auto retry truncated", "Retry on truncation.", "bool"),
            FieldSpec(("inference_params", "suppress_full_response"), "Suppress full response", "Hide full response payload.", "bool"),
            FieldSpec(("inference_params", "results_as_user_role"), "Results as user role", "Emit tool results as user role.", "bool"),
            FieldSpec(("inference_params", "pack_results_as_one_role"), "Pack results as one role", "Pack tool results into one role.", "bool"),
            FieldSpec(("inference_params", "advertised_tools"), "Advertised tools", "List of advertised tools. Supports @file input.", "list"),
            FieldSpec(("inference_params", "silent_tools"), "Silent tools", "List of silent tools. Supports @file input.", "list"),
            FieldSpec(("inference_params", "disabled_tools"), "Disabled tools", "List of disabled tools. Supports @file input.", "list"),
            FieldSpec(("inference_params", "auto_tool_retry_limit"), "Auto tool retry limit", "Tool retry limit.", "int"),
            FieldSpec(("inference_params", "auto_continue_retry_limit"), "Auto continue retry limit", "Continue retry limit.", "int"),
            FieldSpec(("inference_params", "global_tools_mode"), "Global tools mode", "Tools mode for the session.", "enum", ["advertised", "silent", "disabled", "off"]),
        ],
    ),
    (
        "training_params",
        "Training params",
        [
            FieldSpec(("training_params", "training_steps"), "Training steps", "Number of training steps.", "int"),
            FieldSpec(("training_params", "trainer_precision"), "Trainer precision", "Trainer precision.", "enum", ["bf16", "fp16", "fp32"]),
            FieldSpec(("training_params", "lora_r"), "LoRA r", "LoRA rank.", "int"),
            FieldSpec(("training_params", "lora_alpha"), "LoRA alpha", "LoRA alpha.", "int"),
            FieldSpec(("training_params", "lora_dropout"), "LoRA dropout", "LoRA dropout.", "float"),
            FieldSpec(("training_params", "lora_target_modules"), "LoRA target modules", "Comma list or JSON list.", "list"),
            FieldSpec(("training_params", "train_override_ctx"), "Train override ctx", "Override context size.", "text"),
        ],
    ),
]

FILE_INPUT_FIELDS = {
    ("engine_params", "custom_chat_template"),
    ("engine_params", "tools_parser_profile_key"),
    ("engine_params", "default_system_message"),
    ("inference_params", "generation_config_template"),
    ("inference_params", "advertised_tools"),
    ("inference_params", "silent_tools"),
    ("inference_params", "disabled_tools"),
}


def _field_spec_map() -> Dict[Tuple[str, ...], FieldSpec]:
    mapping: Dict[Tuple[str, ...], FieldSpec] = {}
    for _, _, fields in SECTION_SPECS:
        for field in fields:
            mapping[field.path] = field
    return mapping


def _flatten_config(data: Dict[str, Any], prefix: Optional[Tuple[str, ...]] = None) -> Dict[Tuple[str, ...], Any]:
    flat: Dict[Tuple[str, ...], Any] = {}
    current_prefix = prefix or ()
    if not isinstance(data, dict):
        return flat
    for key, value in data.items():
        path = current_prefix + (key,)
        if isinstance(value, dict):
            flat.update(_flatten_config(value, path))
        else:
            flat[path] = value
    return flat


def _print_change_summary(before: Dict[str, Any], after: Dict[str, Any]) -> None:
    before_flat = _flatten_config(before)
    after_flat = _flatten_config(after)
    all_keys = sorted(set(before_flat.keys()) | set(after_flat.keys()))
    changes: List[Tuple[Tuple[str, ...], Any, Any]] = []
    for key in all_keys:
        b = before_flat.get(key, None)
        a = after_flat.get(key, None)
        if b != a:
            changes.append((key, b, a))
    if not changes:
        print("No changes to save.")
        return
    print("Changes to save:")
    for key, b, a in changes:
        print(f"- {'.'.join(key)}: {b!r} -> {a!r}")


def _format_value(value: Any, *, is_set: bool, default_value: Any) -> str:
    if is_set:
        return json.dumps(value) if isinstance(value, (dict, list)) else str(value)
    return _format_hint(default_value)


def _format_hint(value: Any) -> str:
    if value == "":
        return '""'
    if isinstance(value, (dict, list)):
        return json.dumps(value)
    return str(value)


def _parse_value(raw: str, field: FieldSpec) -> Any:
    value = raw.strip()
    if value == "":
        raise ValueError("Empty value not allowed.")
    # quantize_bits must remain a string for engine schema compatibility.
    # Treat "none"/"null" inputs as the literal mode "none".
    if field.path == ("engine_params", "quantize_bits") and value.lower() in {"none", "null"}:
        return "none"
    if value.lower() in {"none", "null"}:
        return None
    if field.kind == "bool":
        if value.lower() in {"true", "1", "yes", "y"}:
            return True
        if value.lower() in {"false", "0", "no", "n"}:
            return False
        raise ValueError("Expected True/False.")
    if field.kind == "int":
        if value.isdigit():
            return int(value)
        raise ValueError("Expected integer.")
    if field.kind == "optional_int":
        parsed = _parse_optional_int(value)
        if value.lower() in {"none", "null"} or parsed is not None:
            return parsed
        raise ValueError("Expected integer or None.")
    if field.kind == "float":
        try:
            return float(value)
        except ValueError as exc:
            raise ValueError("Expected float.") from exc
    if field.kind == "enum":
        if value.lower() in {"none", "null"}:
            return None
        if field.allowed and value not in field.allowed and not field.allow_unknown:
            raise ValueError(f"Expected one of: {', '.join(field.allowed)}.")
        return value
    if field.kind in {"list", "json"}:
        if value.startswith("[") or value.startswith("{"):
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError as exc:
                try:
                    parsed = ast.literal_eval(value)
                except (ValueError, SyntaxError) as exc2:
                    raise ValueError("Invalid JSON input.") from exc2
            return parsed
        if field.kind == "list":
            return [item.strip() for item in value.split(",") if item.strip()]
        return value
    if field.kind == "json_or_text":
        if value.startswith("{") or value.startswith("["):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                try:
                    return ast.literal_eval(value)
                except (ValueError, SyntaxError) as exc:
                    raise ValueError("Invalid JSON input.") from exc
        return value
    return value


def _confirm_unknown_enum_value(field: FieldSpec, value: str) -> bool:
    if field.kind != "enum":
        return True
    if not field.allowed:
        return True
    if value in field.allowed:
        return True
    if not field.allow_unknown:
        return False
    print(
        f"Warning: '{value}' is not in known values for {'.'.join(field.path)} "
        f"({', '.join(field.allowed)})."
    )
    while True:
        decision = input("Accept this value anyway? (y/N): ").strip().lower()
        if decision in {"y", "yes"}:
            return True
        if decision in {"", "n", "no"}:
            return False
        print("Enter y or n.")


def _build_path_resolver(save_path: Path) -> PathResolver:
    cwd = Path.cwd().resolve()
    config_dir = save_path.parent.resolve()
    project_root = detect_project_root(cwd)
    resolver = PathResolver(
        cwd=cwd,
        config_dir=config_dir,
        home_dir=Path.home().resolve(),
        project_dir=project_root.resolve() if project_root else None,
        category_roots={},
    )
    category_paths = dict(DEFAULT_CATEGORY_DIRS)
    resolved_paths: Dict[str, Path] = {}
    for key, value in category_paths.items():
        resolved = resolver.resolve(value, category=None)
        if isinstance(resolved, str):
            resolved_paths[key] = Path(resolved)
    resolver.category_roots = {
        "models": resolved_paths.get("models_root_dir", config_dir),
        "adapters": resolved_paths.get("adapters_root_dir", config_dir),
        "sessions": resolved_paths.get("sessions_root_dir", config_dir),
        "data": resolved_paths.get("data_root_dir", config_dir),
        "tools": resolved_paths.get("tools_root_dir", config_dir),
        "logs": resolved_paths.get("logs_root_dir", config_dir),
    }
    return resolver


def _validate_file_reference(raw: str, *, field: FieldSpec, resolver: PathResolver) -> Optional[str]:
    if not raw.startswith("@"):
        return None
    if field.path not in FILE_INPUT_FIELDS:
        return None
    resolved = resolver.resolve(raw, category=None)
    if not isinstance(resolved, str):
        raise ValueError("Invalid file path.")
    candidate = Path(resolved)
    if not candidate.exists() or not candidate.is_file():
        raise ValueError(f"File not found: {candidate}")
    try:
        candidate.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"Unable to read file: {candidate}") from exc
    return raw


def _interactive_config(
    source_path: Path,
    save_path: Path,
    *,
    existing: dict,
    defaults: dict,
) -> Tuple[dict, Path, bool]:
    _print_intro()
    print(f"Source config path: {source_path}")
    print(f"Save config path: {save_path}")

    config = json.loads(json.dumps(existing))
    original = json.loads(json.dumps(existing))
    resolver = _build_path_resolver(save_path)

    while True:
        print("\nSelect a section:")
        for idx, (_, label, _) in enumerate(SECTION_SPECS, start=1):
            print(f"{idx}. {label}")
        choice = input("Section # (enter=stay, !=save+quit, ^=quit): ").strip()
        if choice == "":
            continue
        if choice == "^":
            return {}, save_path, False
        if choice == "!":
            _print_change_summary(original, config)
            return config, save_path, True
        if not choice.isdigit():
            print("Enter a valid number.")
            continue
        choice_idx = int(choice)
        if not (1 <= choice_idx <= len(SECTION_SPECS)):
            print("Enter a valid number.")
            continue

        section_key, section_label, fields = SECTION_SPECS[choice_idx - 1]
        while True:
            index_width = len(str(len(fields)))
            label_width = max(len(field.label) for field in fields) if fields else 0
            value_width = max(
                len(_format_value(get_nested_value(config, field.path), is_set=_has_nested(config, field.path), default_value=get_nested_value(defaults, field.path)))
                for field in fields
            ) if fields else 0
            print(f"\n{section_label}:")
            for idx, field in enumerate(fields, start=1):
                current_value = get_nested_value(config, field.path)
                default_value = get_nested_value(defaults, field.path)
                is_set = _has_nested(config, field.path)
                display = _format_value(current_value, is_set=is_set, default_value=default_value)
                suffix = " [default]" if not is_set else ""
                allowed_text = f" (allowed: {', '.join(field.allowed)})" if field.allowed else ""
                print(
                    f"{str(idx).rjust(index_width)}. {field.label.ljust(label_width)} = "
                    f"{display.ljust(value_width)}{suffix} — {field.description}{allowed_text}"
                )
            print("-" * max(10, index_width + label_width + value_width + 10))
            raw = input("Field # (enter=back): ").strip()
            if raw == "":
                break
            if raw == "^":
                return {}, save_path, False
            if raw == "!":
                _print_change_summary(original, config)
                return config, save_path, True
            if not raw.isdigit():
                print("Enter a valid number.")
                continue
            field_idx = int(raw)
            if not (1 <= field_idx <= len(fields)):
                print("Enter a valid number.")
                continue

            field = fields[field_idx - 1]
            current_value = get_nested_value(config, field.path)
            default_value = get_nested_value(defaults, field.path)
            is_set = _has_nested(config, field.path)
            display = _format_value(current_value, is_set=is_set, default_value=default_value)
            print(f"\n{str(field_idx).rjust(index_width)}. {'.'.join(field.path)} = {display} — {field.description}")
            if field.allowed:
                print(f"Acceptable values: {', '.join(field.allowed)}")
            while True:
                new_value = input("New value or cancel (#=reset): ").strip()
                if new_value == "":
                    print("Canceled.")
                    break
                if new_value == "^":
                    return {}, save_path, False
                if new_value == "!":
                    _print_change_summary(original, config)
                    return config, save_path, True
                if new_value == "#":
                    delete_nested_value(config, field.path)
                    print("Reset to default.")
                    break
                if field.path == ("engine_params", "default_system_message"):
                    if new_value == "<def>":
                        delete_nested_value(config, field.path)
                        print("Reset to default.")
                        break
                    if new_value == "<>":
                        delete_nested_value(config, field.path)
                        print("Omitted (will use default).")
                        break
                    if new_value == '""':
                        set_nested_value(config, field.path, "")
                        print("Updated.")
                        break
                try:
                    _validate_file_reference(new_value, field=field, resolver=resolver)
                except ValueError as exc:
                    print(str(exc))
                    continue
                try:
                    parsed = _parse_value(new_value, field)
                except ValueError as exc:
                    print(str(exc))
                    continue
                if isinstance(parsed, str) and not _confirm_unknown_enum_value(field, parsed):
                    print("Canceled.")
                    continue
                set_nested_value(config, field.path, parsed)
                print("Updated.")
                break


def main() -> int:
    parser = argparse.ArgumentParser(description="MP13 config helper")
    parser.add_argument("--config", type=str, default=None, help="Target config file path or name.")
    parser.add_argument("--clone", type=str, default=None, help="Clone from another config path or name ('default' uses the default config).")
    parser.add_argument("--init", action="store_true", help="Create a default config file at the target path.")
    parser.add_argument("--interactive", action="store_true", help="Run an interactive config wizard.")
    parser.add_argument("--interractive", dest="interactive", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--reconfigure", nargs="*", metavar="KEY=VALUE", help="Set/reset config values by key path (dot notation).")
    parser.add_argument("--merge", type=str, default=None, help="Merge missing values from a base config into the target config.")
    parser.add_argument("--diff", type=str, default=None, help="Diff the target config against a base config.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing config when used with --init.")
    parser.add_argument("--print", dest="print_config", action="store_true", help="Print resolved config as JSON.")
    parser.add_argument("--print-raw", action="store_true", help="Print raw config file content without resolution.")
    parser.add_argument("--print-set", action="store_true", help="Print only keys set in the config file (no defaults).")
    parser.add_argument("--host-control-state-file", type=str, default=None, help="Path to engine_host_control.json for host auth management.")
    parser.add_argument("--host-auth-status", action="store_true", help="Print host auth status from control state.")
    parser.add_argument("--host-auth-list-keys", action="store_true", help="List host auth keys (without secrets).")
    parser.add_argument("--host-auth-generate-secret", type=int, default=0, metavar="BYTES", help="Generate a random key secret token (BYTES entropy hint).")
    parser.add_argument("--host-auth-upsert-key", type=str, default=None, metavar="KEY_ID", help="Create/update host auth key.")
    parser.add_argument(
        "--host-auth-role",
        type=str,
        default=None,
        help=(
            "Role for --host-auth-upsert-key: "
            "admin|config_editor|worker_user|model_user_with_model_control|model_user|diagnostic_user."
        ),
    )
    parser.add_argument("--host-auth-disable-key", action="store_true", help="Mark upserted key as disabled.")
    parser.add_argument("--host-auth-revoke-key", type=str, default=None, metavar="KEY_ID", help="Revoke host auth key by ID.")
    parser.add_argument("--host-auth-issue-session", type=str, default=None, metavar="KEY_ID", help="Issue auth session token using key ID + secret.")
    parser.add_argument("--host-auth-scope", type=str, default="control", help="Session scope: control|config|traffic.")
    parser.add_argument("--host-auth-ttl-seconds", type=int, default=900, help="Session TTL seconds.")
    parser.add_argument("--host-auth-config-paths", type=str, default="", help="Comma-separated allowed config selectors for config scope.")
    parser.add_argument("--host-auth-engine-ids", type=str, default="", help="Comma-separated allowed engine IDs for traffic scope.")
    parser.add_argument("--host-auth-revoke-session", type=str, default=None, metavar="TOKEN", help="Revoke session token.")
    parser.add_argument("--host-auth-require-auth", type=str, default=None, help="Set require_auth in host control config (true/false).")
    parser.add_argument("--host-auth-allowed-configs", type=str, default="", help="Comma-separated allowed configs for config_editor role key.")
    parser.add_argument(
        "--host-auth-allowed-engines",
        type=str,
        default="",
        help="Comma-separated allowed engine IDs for worker/model role keys.",
    )
    parser.add_argument("--host-auth-secret", type=str, default="", help="Key secret (not recommended in shell history).")
    parser.add_argument("--host-auth-secret-env", type=str, default="", help="Environment variable name holding key secret.")
    parser.add_argument("--host-auth-secret-stdin", action="store_true", help="Read key secret from stdin (single line).")
    args = parser.parse_args()

    if isinstance(args.host_auth_require_auth, str):
        raw = str(args.host_auth_require_auth).strip().lower()
        if raw in {"1", "true", "yes", "on"}:
            args.host_auth_require_auth = True
        elif raw in {"0", "false", "no", "off"}:
            args.host_auth_require_auth = False
        else:
            print("Invalid --host-auth-require-auth value. Use true/false.")
            return 1

    host_auth_rc = _run_host_auth_ops(args)
    if host_auth_rc is not None:
        return int(host_auth_rc)

    cwd = Path.cwd()
    target_path = _resolve_target_path(args, cwd=cwd)
    default_path = get_default_config_path()

    if args.clone:
        source_path = resolve_config_path(args.clone, cwd=cwd, default_config_path=default_path)
    else:
        source_path = target_path

    if args.clone and not (args.interactive or args.init or args.print_config or args.print_raw or args.print_set or args.merge or args.diff):
        args.interactive = True

    if args.diff:
        base_path = resolve_config_path(args.diff, cwd=cwd, default_config_path=default_path)
        base_config = _load_config_or_default(base_path, default_path=default_path, defaults=_build_template_config())
        target_config = load_json_config(target_path) or {}
        if base_path == default_path:
            diff = strip_defaults(target_config, base_config)
        else:
            diff = diff_configs(base_config, target_config)
        print(json.dumps(diff, indent=2))
        return 0

    if args.merge and not (args.print_config or args.print_raw or args.print_set or args.interactive or args.init or args.reconfigure):
        base_path = resolve_config_path(args.merge, cwd=cwd, default_config_path=default_path)
        base_config = _load_config_or_default(base_path, default_path=default_path, defaults=_build_template_config())
        target_config = load_json_config(target_path) or {}
        merged = deep_merge_dicts(base_config, target_config)
        save_json_config(merged, target_path)
        print(f"Config merged with {base_path} at {target_path}")
        return 0

    if args.reconfigure is not None:
        updates = args.reconfigure
        if not updates:
            print("Provide KEY=VALUE pairs for --reconfigure.")
            print("Valid keys include:")
            field_map = _field_spec_map()
            for key in sorted(".".join(path) for path in field_map.keys()):
                print(f"  - {key}")
            return 1
        config = load_json_config(target_path) or {}
        resolver = _build_path_resolver(target_path)
        field_map = _field_spec_map()
        for item in updates:
            if "=" not in item:
                print(f"Invalid update '{item}'. Use KEY=VALUE.")
                continue
            key, raw_value = item.split("=", 1)
            key_path = tuple(part.strip() for part in key.strip().split(".") if part.strip())
            if not key_path:
                print(f"Invalid key in '{item}'.")
                continue
            field_spec = field_map.get(key_path)
            if field_spec is None:
                print(f"Warning: Unknown key '{'.'.join(key_path)}'. Skipping.")
                continue
            value_text = raw_value.strip()
            if value_text == "" or value_text.lower() in {"reset", "none", "null", "undefined"}:
                delete_nested_value(config, key_path)
                continue
            try:
                _validate_file_reference(value_text, field=field_spec, resolver=resolver)
            except ValueError as exc:
                print(f"Warning: {'.'.join(key_path)}: {exc}. Skipping.")
                continue
            try:
                value = _parse_value(value_text, field_spec)
            except ValueError as exc:
                print(f"Warning: {'.'.join(key_path)}: {exc}. Skipping.")
                continue
            set_nested_value(config, key_path, value)
        save_json_config(config, target_path)
        print(f"Config updated at {target_path}")
        return 0

    if args.interactive:
        defaults = _build_template_config()
        existing = load_json_config(source_path) or {}
        wizard_config, final_path, should_save = _interactive_config(
            source_path,
            target_path,
            existing=existing,
            defaults=defaults,
        )
        if should_save and wizard_config is not None:
            save_json_config(wizard_config, final_path)
            resolved, _ = resolve_config_paths(wizard_config, cwd=cwd, config_path=final_path)
            _ensure_category_dirs(resolved)
            print(f"Config written to {final_path}")
        return 0

    if args.init:
        if target_path.exists() and not args.force:
            print(f"Config already exists at {target_path}. Use --force to overwrite.")
            return 1
        defaults = _build_template_config()
        is_default_target = target_path == default_path
        if args.clone:
            source_cfg = _load_config_or_default(source_path, default_path=default_path, defaults=defaults)
            if args.merge:
                base_path = resolve_config_path(args.merge, cwd=cwd, default_config_path=default_path)
                base_cfg = _load_config_or_default(base_path, default_path=default_path, defaults=defaults)
                source_cfg = deep_merge_dicts(base_cfg, source_cfg)
            save_json_config(source_cfg, target_path)
            resolved, _ = resolve_config_paths(source_cfg, cwd=cwd, config_path=target_path)
            _ensure_category_dirs(resolved)
            print(f"Config written to {target_path}")
            return 0
        template = defaults if is_default_target else {}
        if args.merge:
            base_path = resolve_config_path(args.merge, cwd=cwd, default_config_path=default_path)
            base_cfg = _load_config_or_default(base_path, default_path=default_path, defaults=defaults)
            template = deep_merge_dicts(base_cfg, template)
        save_json_config(template, target_path)
        resolved, _ = resolve_config_paths(template, cwd=cwd, config_path=target_path)
        _ensure_category_dirs(resolved)
        print(f"Config written to {target_path}")
        return 0

    if args.print_raw:
        raw = load_json_config(target_path) or {}
        print(json.dumps(raw, indent=2))
        return 0

    if args.print_set:
        raw = load_json_config(target_path) or {}
        print(json.dumps(raw, indent=2))
        return 0

    if args.print_config:
        config = load_json_config(target_path) or {}
        resolved, _ = resolve_config_paths(
            config,
            cwd=cwd,
            config_path=target_path,
        )
        print(json.dumps(resolved, indent=2))
        return 0

    if not args.init and not args.print_config:
        print(f"Target config path: {target_path}")
        print("Use --init to create or --print to resolve.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


def cli() -> int:
    return main()
