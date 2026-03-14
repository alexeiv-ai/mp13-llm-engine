# Copyright (c) 2025 mp13
# Author: alexeiv-ai <188820640+alexeiv-ai@users.noreply.github.com>
# AI-Assistance: Portions of this file were drafted using AI coding tools
# (e.g., ChatGPT, Gemini, Codex) under active human design supervision.
# Contact: Please open an issue or discussion on GitHub.
# SPDX-License-Identifier: Apache-2.0
"""MP13 server package - Unified training and inference server.

This package entrypoint intentionally avoids importing heavy runtime modules
at import time so utility submodules (for example config path helpers) remain
cheap to import.
"""

from __future__ import annotations

import importlib
import logging as _logging
import warnings
from typing import Dict, Tuple

_logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(_logging.ERROR)
warnings.filterwarnings("ignore", category=SyntaxWarning)

_LAZY_EXPORTS: Dict[str, Tuple[str, str]] = {
    # Config classes
    "APIStatus": ("mp13_config", "APIStatus"),
    "GlobalEngineConfig": ("mp13_config", "GlobalEngineConfig"),
    "TrainingConfig": ("mp13_config", "TrainingConfig"),
    "InferenceConfig": ("mp13_config", "InferenceConfig"),
    "InferenceRequest": ("mp13_config", "InferenceRequest"),
    "InferenceResponse": ("mp13_config", "InferenceResponse"),
    "DatasetFormat": ("mp13_config", "DatasetFormat"),
    "ColumnsConfig": ("mp13_config", "ColumnsConfig"),
    "TrainingMode": ("mp13_config", "TrainingMode"),
    "EngineMode": ("mp13_config", "EngineMode"),
    "AdapterConfig": ("mp13_config", "AdapterConfig"),
    "AdapterType": ("mp13_config", "AdapterType"),
    "DatasetConfig": ("mp13_config", "DatasetConfig"),
    "DatasetTags": ("mp13_config", "DatasetTags"),
    "PreprocessingMode": ("mp13_config", "PreprocessingMode"),
    # State classes
    "MP13State": ("mp13_state", "MP13State"),
    "TrainingStatus": ("mp13_state", "TrainingStatus"),
    "InferenceStatus": ("mp13_state", "InferenceStatus"),
    "ServerStatus": ("mp13_state", "ServerStatus"),
    # Error classes
    "ConfigurationError": ("mp13_errors", "ConfigurationError"),
    "DatasetError": ("mp13_errors", "DatasetError"),
    "TrainingError": ("mp13_errors", "TrainingError"),
    "EngineError": ("mp13_errors", "EngineError"),
    "EngineInitializationError": ("mp13_errors", "EngineInitializationError"),
    "AdapterError": ("mp13_errors", "AdapterError"),
    "InferenceRequestError": ("mp13_errors", "InferenceRequestError"),
    "BusyError": ("mp13_errors", "BusyError"),
    # API and engine runtime
    "handle_call_tool": ("mp13_engine_api", "handle_call_tool"),
    "MP13Engine": ("mp13_engine", "MP13Engine"),
    "logger": ("mp13_engine", "logger"),
}

__all__ = list(_LAZY_EXPORTS.keys())


def __getattr__(name: str):
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = target
    module = importlib.import_module(f".{module_name}", __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals().keys()) | set(__all__))
