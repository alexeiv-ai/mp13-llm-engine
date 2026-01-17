from __future__ import annotations

import sys
import importlib.util
from pathlib import Path


_SRC_DIR = Path(__file__).resolve().parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))


def _load_config_main():
    module_path = _SRC_DIR / "app" / "config.py"
    spec = importlib.util.spec_from_file_location("mp13_app_config", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load config module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main


main = _load_config_main()


if __name__ == "__main__":
    raise SystemExit(main())
