# Copyright (c) 2025 mp13
# Author: alexeiv-ai <188820640+alexeiv-ai@users.noreply.github.com>
# AI-Assistance: Portions of this file were drafted using AI coding tools
# (e.g., ChatGPT, Gemini, Codex) under active human design supervision.
# Contact: Please open an issue or discussion on GitHub.
# SPDX-License-Identifier: Apache-2.0
"""MP13 server package - Unified training and inference server."""

import logging as _logging
_logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(_logging.ERROR)

from .mp13_config import (
    APIStatus, GlobalEngineConfig, TrainingConfig, InferenceConfig, InferenceRequest, 
    InferenceResponse, DatasetFormat, ColumnsConfig, TrainingMode,
    EngineMode, AdapterConfig, AdapterType, 
    DatasetConfig, DatasetTags, PreprocessingMode 
)
from .mp13_state import (
    MP13State, TrainingStatus, InferenceStatus, ServerStatus,
    ConfigurationError, DatasetError, TrainingError, EngineError, 
    EngineInitializationError, AdapterError, InferenceRequestError, BusyError
)
from .mp13_engine_api import handle_call_tool
from .mp13_engine  import MP13Engine
from .mp13_engine import logger as logger

# Import PEFT classes but don't re-export them directly
# This avoids type conflicts
import peft

# You can decide which components to make directly available
__all__ = [
    # Config classes
    "APIStatus", "GlobalEngineConfig", "TrainingConfig", "InferenceConfig",
    "InferenceRequest", "InferenceResponse",
    "TrainingMode", "DatasetFormat", 
    
    # State classes
    "MP13State", "TrainingStatus", "InferenceStatus", "ServerStatus",
    
    # Error classes
    "ConfigurationError", "DatasetError", "TrainingError", "EngineError",
    "EngineInitializationError", "AdapterError", "InferenceRequestError", "BusyError",
    
    # Main API
    "handle_call_tool",
    
    # Engine class
    "MP13Engine", "logger",
    
]
