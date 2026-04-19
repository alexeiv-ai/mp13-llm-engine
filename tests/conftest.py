from __future__ import annotations

import sys
import os
import tempfile
from pathlib import Path


def _ensure_src_on_path() -> None:
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def _ensure_pytest_temp_root() -> None:
    # Keep pytest temp artifacts outside the repo root by default.
    root = Path(__file__).resolve().parents[1]
    candidates = [
        root.parent / ".mp13_pytest",
        Path(tempfile.gettempdir()) / "mp13_pytest",
        root / ".tmp_pytest",
    ]
    for base in candidates:
        try:
            base.mkdir(parents=True, exist_ok=True)
            os.environ.setdefault("PYTEST_DEBUG_TEMPROOT", str(base))
            return
        except PermissionError:
            continue


_ensure_src_on_path()
_ensure_pytest_temp_root()
