# Copyright (c) 2025 mp13
# Author: alexeiv-ai <188820640+alexeiv-ai@users.noreply.github.com>
# AI-Assistance: Portions of this file were drafted using AI coding tools
# (e.g., ChatGPT, Gemini, Codex) under active human design supervision.
# Contact: Please open an issue or discussion on GitHub.
# SPDX-License-Identifier: Apache-2.0
"""Lightweight shared error taxonomy for MP13."""

class MP13Error(Exception):
    pass


class ConfigurationError(MP13Error):
    pass


class DatasetError(MP13Error):
    pass


class TrainingError(MP13Error):
    pass


class EngineError(MP13Error):
    pass


class EngineInitializationError(EngineError):
    pass


class AdapterError(EngineError):
    pass


class InferenceRequestError(EngineError):
    pass


class BusyError(MP13Error):
    pass


class ModeMismatchError(EngineError):
    pass
