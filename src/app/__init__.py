# Copyright (c) 2025 mp13
# Author: alexeiv-ai <188820640+alexeiv-ai@users.noreply.github.com>
# AI-Assistance: Portions of this file were drafted using AI coding tools
# (e.g., ChatGPT, Gemini, Codex) under active human design supervision.
# Contact: Please open an issue or discussion on GitHub.
# SPDX-License-Identifier: MIT
"""App-layer APIs and CLI helpers for MP13."""

import warnings
import sys
from pathlib import Path

warnings.filterwarnings("ignore", category=SyntaxWarning)


# Allow `python -m src.app.*` without an install by exposing `src/` on sys.path.
_SRC_ROOT = Path(__file__).resolve().parents[1]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from .engine_session import (
    Colors,
    Command,
    ChatSession,
    EngineSession,
    InferenceParams,
    ReentrantWriterFairRWLock,
    Turn,
)
from .context_cursor import (
    ChatContext,
    ChatContextScope,
    ChatCursor,
    ChatForks,
    StreamDisplayContext,
    StreamDisplayPlan,
)

__all__ = [
    "Colors",
    "Command",
    "ChatSession",
    "EngineSession",
    "InferenceParams",
    "ReentrantWriterFairRWLock",
    "Turn",
    "ChatContext",
    "ChatContextScope",
    "ChatCursor",
    "ChatForks",
    "StreamDisplayContext",
    "StreamDisplayPlan",
]
