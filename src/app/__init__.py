# Copyright (c) 2025 mp13
# Author: alexeiv-ai <188820640+alexeiv-ai@users.noreply.github.com>
# AI-Assistance: Portions of this file were drafted using AI coding tools
# (e.g., ChatGPT, Gemini, Codex) under active human design supervision.
# Contact: Please open an issue or discussion on GitHub.
# SPDX-License-Identifier: MIT
"""App-layer APIs and CLI helpers for MP13."""

from __future__ import annotations

import warnings
import sys
import importlib
from pathlib import Path
from typing import Dict, Tuple

warnings.filterwarnings("ignore", category=SyntaxWarning)


# Allow `python -m src.app.*` without an install by exposing `src/` on sys.path.
_SRC_ROOT = Path(__file__).resolve().parents[1]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

_LAZY_EXPORTS: Dict[str, Tuple[str, str]] = {
    "Colors": ("engine_session", "Colors"),
    "Command": ("engine_session", "Command"),
    "ChatSession": ("engine_session", "ChatSession"),
    "EngineSession": ("engine_session", "EngineSession"),
    "InferenceParams": ("engine_session", "InferenceParams"),
    "ReentrantWriterFairRWLock": ("engine_session", "ReentrantWriterFairRWLock"),
    "Turn": ("engine_session", "Turn"),
    "ChatContext": ("context_cursor", "ChatContext"),
    "ChatContextScope": ("context_cursor", "ChatContextScope"),
    "ChatCursor": ("context_cursor", "ChatCursor"),
    "ChatForks": ("context_cursor", "ChatForks"),
    "StreamDisplayContext": ("context_cursor", "StreamDisplayContext"),
    "StreamDisplayPlan": ("context_cursor", "StreamDisplayPlan"),
}

__all__ = [
    *list(_LAZY_EXPORTS.keys()),
]


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
