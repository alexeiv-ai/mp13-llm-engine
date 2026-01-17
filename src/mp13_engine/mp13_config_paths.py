# Copyright (c) 2025 mp13
# Author: alexeiv-ai <188820640+alexeiv-ai@users.noreply.github.com>
# AI-Assistance: Portions of this file were drafted using AI coding tools
# (e.g., ChatGPT, Gemini, Codex) under active human design supervision.
# Contact: Please open an issue or discussion on GitHub.
# SPDX-License-Identifier: Apache-2.0
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


APP_DIR_NAME = ".mp13-llm"
DEFAULT_CONFIG_FILENAME = "mp13_config.json"

DEFAULT_CATEGORY_DIRS = {
    "models_root_dir": "@project/..",
    "adapters_root_dir": "@project/adapters",
    "data_root_dir": "@project/data",
    "sessions_root_dir": "@home/.mp13-llm/sessions",
    "tools_root_dir": "@project/configs",
    "logs_root_dir": "@temp",
}

DEFAULT_INFERENCE_PARAMS = {
    "stream": True,
    "cache": None,
    "return_prompt": None,
    "generation_config_template": {},
    "max_new_tokens": None,
    "no_tools_parse": False,
    "auto_retry_truncated": False,
    "suppress_full_response": False,
    "results_as_user_role": False,
    "pack_results_as_one_role": False,
    "advertised_tools": [],
    "silent_tools": [],
    "disabled_tools": [],
    "auto_tool_retry_limit": 5,
    "auto_continue_retry_limit": 10,
    "global_tools_mode": "advertised",
}

DEFAULT_TRAINING_PARAMS = {
    "training_steps": 100,
    "trainer_precision": "bf16",
    "lora_r": 8,
    "lora_alpha": 64,
    "lora_dropout": 0.0,
    "lora_target_modules": None,
    "train_override_ctx": "auto",
}

DEFAULT_CHAT_CONFIG = {
    "category_dirs": DEFAULT_CATEGORY_DIRS,
    "engine_params": {
        "base_model_path": "",
        "base_model_dtype": "auto",
        "quantize_bits": "none",
        "hqq_bits": 4,
        "hqq_group_size": 64,
        "hqq_quant_zero": True,
        "hqq_quant_scale": False,
        "hqq_axis": 1,
        "default_system_message": "",
        "default_context_size": None,
        "default_max_new_tokens": 8192,
        "attn_implementation": "auto",
        "use_cache": True,
        "device_map": "auto",
        "trust_remote_code": True,
        "use_torch_compile": True,
        "static_kv_cache": False,
        "concurrent_generate": 4,
        "tools_config_path": "mp13tools.json",
        "console_log_level": "warning",
        "file_log_level": "debug",
    },
    "inference_params": DEFAULT_INFERENCE_PARAMS,
    "training_params": DEFAULT_TRAINING_PARAMS,
}

CATEGORY_DIRS_KEY = "category_dirs"

CATEGORY_ROOT_KEYS = {
    "models": "models_root_dir",
    "adapters": "adapters_root_dir",
    "sessions": "sessions_root_dir",
    "data": "data_root_dir",
    "tools": "tools_root_dir",
    "logs": "logs_root_dir",
}

TOOLS_CONFIG_KEYS = ("tools_config_path",)
ENGINE_PATH_KEYS = {
    "base_model_path": ("models", False),
    "adapters_root_dir": ("adapters", False),
    "sessions_save_dir": ("sessions", False),
    "data_root_dir": ("data", False),
    "tools_root_dir": ("tools", False),
    "models_root_dir": ("models", False),
    "logs_root_dir": ("logs", False),
}


def get_default_config_dir() -> Path:
    return Path.home() / APP_DIR_NAME


def get_default_config_path() -> Path:
    return get_default_config_dir() / DEFAULT_CONFIG_FILENAME




def detect_project_root(start: Path) -> Optional[Path]:
    current = start.resolve()
    for candidate in [current, *current.parents]:
        if (candidate / ".git").exists():
            return candidate
    return None


def load_json_config(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        with open(path, "r") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        return None
    return None


def save_json_config(config_data: Dict[str, Any], save_to_path: Path) -> None:
    save_to_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_to_path, "w") as f:
        json.dump(config_data, f, indent=2)


def deep_merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def _has_explicit_rel_prefix(value: str) -> bool:
    return value.startswith(("./", ".\\", "../", "..\\"))


def _split_anchor(value: str) -> Tuple[str, str]:
    raw = value[1:]
    if not raw:
        return "", ""
    sep_index = len(raw)
    for sep in ("/", "\\"):
        idx = raw.find(sep)
        if idx != -1:
            sep_index = min(sep_index, idx)
    anchor = raw[:sep_index]
    rest = raw[sep_index + 1:] if sep_index < len(raw) else ""
    return anchor, rest


@dataclass
class PathResolver:
    cwd: Path
    config_dir: Path
    home_dir: Path
    project_dir: Optional[Path]
    category_roots: Dict[str, Path]

    def resolve(self, value: Any, category: Optional[str] = None, *, allow_remote_id: bool = False) -> Any:
        if value is None or not isinstance(value, str):
            return value
        raw = value.strip()
        if not raw:
            return value
        if raw.startswith("@"):
            anchor, rest = _split_anchor(raw)
            base = self._anchor_base(anchor)
            if base is None:
                return raw
            return str((base / rest).resolve()) if rest else str(base.resolve())
        if raw.startswith("~"):
            return str(Path(raw).expanduser().resolve())
        if Path(raw).is_absolute():
            return str(Path(raw).resolve())
        if _has_explicit_rel_prefix(raw):
            return str((self.cwd / raw).resolve())
        if category and category in self.category_roots:
            return str((self.category_roots[category] / raw).resolve())
        return str((self.config_dir / raw).resolve())

    def _anchor_base(self, anchor: str) -> Optional[Path]:
        if anchor == "home":
            return self.home_dir
        if anchor == "temp":
            return Path(tempfile.gettempdir())
        if anchor == "project":
            return self.project_dir or self.cwd
        if anchor == "config":
            return self.config_dir
        if anchor in CATEGORY_ROOT_KEYS:
            return self.category_roots.get(anchor)
        return None


def _build_paths_from_config(config: Dict[str, Any]) -> Dict[str, Any]:
    paths = dict(DEFAULT_CATEGORY_DIRS)
    if isinstance(config.get(CATEGORY_DIRS_KEY), dict):
        paths.update(config[CATEGORY_DIRS_KEY])
    return paths


def resolve_config_paths(
    config: Dict[str, Any],
    *,
    cwd: Optional[Path] = None,
    config_path: Optional[Path] = None,
    project_root: Optional[Path] = None,
) -> Tuple[Dict[str, Any], PathResolver]:
    cwd = (cwd or Path.cwd()).resolve()
    config_path = config_path or get_default_config_path()
    config_dir = config_path.parent.resolve()
    project_root = project_root or detect_project_root(cwd)
    resolver = PathResolver(
        cwd=cwd,
        config_dir=config_dir,
        home_dir=Path.home().resolve(),
        project_dir=project_root.resolve() if project_root else None,
        category_roots={},
    )
    category_paths = dict(DEFAULT_CATEGORY_DIRS)
    if isinstance(config.get(CATEGORY_DIRS_KEY), dict):
        category_paths.update(config[CATEGORY_DIRS_KEY])
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

    config = dict(config)
    config[CATEGORY_DIRS_KEY] = {k: str(v) for k, v in resolved_paths.items()}
    if "adapters_root_dir" in resolved_paths:
        config["adapters_root_dir"] = str(resolver.category_roots["adapters"])
    if "sessions_save_dir" in config or "sessions_root_dir" in resolved_paths:
        config["sessions_save_dir"] = str(resolver.category_roots["sessions"])
    if "data_root_dir" in resolved_paths:
        config["data_root_dir"] = str(resolver.category_roots["data"])
    if "tools_root_dir" in resolved_paths:
        config["tools_root_dir"] = str(resolver.category_roots["tools"])
    if "models_root_dir" in resolved_paths:
        config["models_root_dir"] = str(resolver.category_roots["models"])
    if "logs_root_dir" in resolved_paths:
        config["logs_root_dir"] = str(resolver.category_roots["logs"])

    if "base_model_path" in config:
        config["base_model_path"] = resolver.resolve(
            config["base_model_path"],
            category="models",
            allow_remote_id=False,
        )
    for key in TOOLS_CONFIG_KEYS:
        if key in config:
            config[key] = resolver.resolve(config[key], category="tools")
    if "inference_params" in config and isinstance(config["inference_params"], dict):
        params = dict(config["inference_params"])
        for key in TOOLS_CONFIG_KEYS:
            if key in params:
                params[key] = resolver.resolve(params[key], category="tools")
        config["inference_params"] = params

    return config, resolver


def resolve_engine_inputs(config: Dict[str, Any], resolver: PathResolver) -> Dict[str, Any]:
    resolved = dict(config)
    engine_params = resolved.get("engine_params")
    if isinstance(engine_params, dict):
        flattened = dict(engine_params)
        flattened.update({k: v for k, v in resolved.items() if k != "engine_params"})
        resolved = flattened
    for key, (category, allow_remote) in ENGINE_PATH_KEYS.items():
        if key in resolved:
            resolved[key] = resolver.resolve(resolved[key], category=category, allow_remote_id=allow_remote)
    for key in TOOLS_CONFIG_KEYS:
        if key in resolved:
            resolved[key] = resolver.resolve(resolved[key], category="tools")
    if "dataset_path" in resolved:
        resolved["dataset_path"] = resolver.resolve(resolved["dataset_path"], category="data")
    dataset = resolved.get("dataset")
    if isinstance(dataset, dict) and "dataset_path" in dataset:
        dataset_copy = dict(dataset)
        dataset_copy["dataset_path"] = resolver.resolve(dataset_copy["dataset_path"], category="data")
        resolved["dataset"] = dataset_copy
    return resolved


def load_merged_config(
    *,
    default_config_path: Optional[Path] = None,
    custom_config_path: Optional[Path] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    default_path = default_config_path or get_default_config_path()
    default_config = load_json_config(default_path) or {}
    custom_config = load_json_config(custom_config_path) if custom_config_path else {}
    if custom_config is None:
        custom_config = {}
    merged = deep_merge_dicts(default_config, custom_config)
    return merged, default_config, custom_config


def ensure_required_paths(resolver: PathResolver, required_categories: Tuple[str, ...]) -> Tuple[bool, Dict[str, str]]:
    missing: Dict[str, str] = {}
    for category in required_categories:
        root = resolver.category_roots.get(category)
        if not root:
            missing[category] = ""
            continue
        if not root.exists():
            missing[category] = str(root)
    return (len(missing) == 0), missing


def resolve_custom_config_path(
    config_path: Optional[str],
    default_dir: Optional[Path] = None,
) -> Optional[Path]:
    if not config_path:
        return None
    candidate = Path(config_path)
    if candidate.is_absolute() or str(candidate).startswith(("./", ".\\", "../", "..\\")):
        return candidate.expanduser().resolve()
    base_dir = default_dir or get_default_config_dir()
    return (base_dir / candidate).expanduser().resolve()


def load_effective_config(
    *,
    default_config_path: Optional[Path] = None,
    custom_config_path: Optional[Path] = None,
    cwd: Optional[Path] = None,
) -> Tuple[Optional[Dict[str, Any]], Optional[PathResolver], bool]:
    default_path = default_config_path or get_default_config_path()
    default_config = load_json_config(default_path) or {}
    custom_config = load_json_config(custom_config_path) if custom_config_path else {}
    if custom_config is None:
        custom_config = {}
    if not default_config and not custom_config:
        return None, None, False
    merged = deep_merge_dicts(default_config, custom_config)
    resolved, resolver = resolve_config_paths(
        merged,
        cwd=cwd or Path.cwd(),
        config_path=custom_config_path or default_path,
    )
    resolved = resolve_engine_inputs(resolved, resolver)
    return resolved, resolver, True
