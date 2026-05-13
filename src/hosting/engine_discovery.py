"""Engine discovery helpers."""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from typing import Optional, Tuple

from ._process_utils import hidden_subprocess_kwargs


def _get_target_python(python_executable: Optional[str] = None) -> str:
    if python_executable:
        return python_executable
    return os.environ.get("MP13_ENGINE_PYTHON", "").strip() or sys.executable


def _is_same_python(python_executable: str) -> bool:
    try:
        return os.path.realpath(python_executable) == os.path.realpath(sys.executable)
    except Exception:
        return python_executable == sys.executable


def is_engine_discoverable(python_executable: Optional[str] = None) -> Tuple[bool, str]:
    """
    Lightweight check if 'mp13_engine' is discoverable.
    
    If python_executable is not provided, it respects the MP13_ENGINE_PYTHON
    environment variable, falling back to sys.executable.

    Avoids a subprocess (and console window on Windows) if the target Python
    is the same as the current Python executable.
    """
    target_python = _get_target_python(python_executable)

    if _is_same_python(target_python):
        try:
            found = importlib.util.find_spec("mp13_engine") is not None
            if found:
                return True, ""
            return False, "module not discoverable"
        except Exception as exc:
            return False, str(exc)

    probe = (
        "import importlib.util, sys; "
        "sys.exit(0 if importlib.util.find_spec('mp13_engine') else 1)"
    )

    kwargs = hidden_subprocess_kwargs()

    try:
        result = subprocess.run(  # noqa: S603
            [target_python, "-c", probe],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
            **kwargs
        )
        if result.returncode == 0:
            return True, ""
        stderr = (result.stderr or "").strip()
        last_line = stderr.splitlines()[-1] if stderr else "module not discoverable"
        return False, last_line
    except FileNotFoundError:
        return False, f"Python executable not found: {target_python}"
    except Exception as exc:
        return False, str(exc)


def is_engine_available(python_executable: Optional[str] = None) -> Tuple[bool, str]:
    """
    Strict check whether engine runtime symbols are actually importable.

    This intentionally performs a heavy runtime import and can transitively load
    ML dependencies such as torch. Do not use it for daemon hot paths or spawn
    preflight; use is_engine_discoverable there and let worker startup be the
    authoritative runtime check. This is still useful for explicit diagnostics
    and setup validation where proving MP13Engine imports is the requested work.
    
    If python_executable is not provided, it respects the MP13_ENGINE_PYTHON
    environment variable, falling back to sys.executable.

    Avoids a subprocess (and console window on Windows) if the target Python
    is the same as the current Python executable.
    """
    target_python = _get_target_python(python_executable)

    if _is_same_python(target_python):
        try:
            import mp13_engine  # type: ignore
            # Check for the expected symbol
            _ = getattr(mp13_engine, "MP13Engine", None)
            if _ is None:
                return False, "module found but MP13Engine symbol is missing"
            return True, ""
        except ImportError as exc:
            return False, str(exc)
        except Exception as exc:
            return False, str(exc)

    kwargs = hidden_subprocess_kwargs()

    try:
        result = subprocess.run(  # noqa: S603
            [target_python, "-c", "from mp13_engine import MP13Engine"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
            **kwargs
        )
        if result.returncode == 0:
            return True, ""
        stderr = (result.stderr or "").strip()
        last_line = stderr.splitlines()[-1] if stderr else "import failed"
        return False, last_line
    except FileNotFoundError:
        return False, f"Python executable not found: {target_python}"
    except subprocess.TimeoutExpired:
        return False, "strict import timed out after 30 seconds"
    except Exception as exc:
        return False, str(exc)
