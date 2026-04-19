from __future__ import annotations

import importlib
import sys


def test_importing_config_paths_does_not_load_heavy_engine_modules() -> None:
    # Reset relevant modules so this test observes a fresh import path.
    for name in [
        "mp13_engine",
        "mp13_engine.mp13_config_paths",
        "mp13_engine.mp13_engine",
        "mp13_engine.mp13_state",
        "mp13_engine.mp13_engine_api",
    ]:
        sys.modules.pop(name, None)

    mod = importlib.import_module("mp13_engine.mp13_config_paths")
    assert mod is not None
    assert "mp13_engine.mp13_engine" not in sys.modules
    assert "mp13_engine.mp13_state" not in sys.modules
    assert "mp13_engine.mp13_engine_api" not in sys.modules
