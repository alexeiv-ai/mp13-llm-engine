"""Fixed-location daemon lifecycle and crash diagnostics."""
from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

_REPORT_HANDLE: Optional[Any] = None
_REPORT_PATH: Optional[Path] = None
_INSTALLED = False
_LOCK = threading.RLock()


def default_daemon_report_path() -> Path:
    try:
        from mp13_engine.mp13_config_paths import get_hosting_root_dir

        return (get_hosting_root_dir() / "logs" / "daemon-crash.log").expanduser().resolve()
    except Exception:
        return (Path.home() / ".mp13-llm" / "hosting" / "logs" / "daemon-crash.log").expanduser().resolve()


def daemon_report_path_for_control_state(control_state_file: Optional[Path]) -> Path:
    if control_state_file is None:
        return default_daemon_report_path()
    raw = Path(control_state_file).expanduser().resolve()
    hosting_root = raw.parent if raw.suffix else raw
    return (hosting_root / "logs" / "daemon-crash.log").resolve()


def _timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())


def _render_report(
    *,
    event: str,
    reason: str,
    actor: Optional[Dict[str, Any]] = None,
    details: Optional[Dict[str, Any]] = None,
) -> str:
    payload = {
        "event": str(event or "daemon_event"),
        "timestamp_utc": _timestamp(),
        "timestamp_unix": time.time(),
        "pid": None,
        "reason": str(reason or "unknown"),
        "actor": dict(actor or {}),
        "details": dict(details or {}),
        "python": sys.version,
        "platform": sys.platform,
        "argv": list(sys.argv),
    }
    try:
        import os

        payload["pid"] = os.getpid()
    except Exception:
        pass
    return (
        "MP13 hosting daemon diagnostic report\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
        + "\n"
    )


def write_daemon_report(
    *,
    event: str,
    reason: str,
    actor: Optional[Dict[str, Any]] = None,
    details: Optional[Dict[str, Any]] = None,
    path: Optional[Path] = None,
    overwrite: bool = True,
) -> Path:
    """Write the fixed daemon report file, overwriting by default."""
    path = (path.expanduser().resolve() if path is not None else (_REPORT_PATH or default_daemon_report_path()))
    text = _render_report(event=event, reason=reason, actor=actor, details=details)
    with _LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        fp = _REPORT_HANDLE
        if overwrite and fp is not None and not fp.closed:
            try:
                fp.seek(0)
                fp.truncate(0)
                fp.write(text)
                fp.flush()
                return path
            except Exception:
                pass
        mode = "w" if overwrite else "a"
        with path.open(mode, encoding="utf-8") as out:
            if not overwrite:
                out.write("\n" + "=" * 80 + "\n")
            out.write(text)
    return path


def install_daemon_crash_report(path: Optional[Path] = None) -> Path:
    """
    Install best-effort crash diagnostics for daemon entrypoints.

    This catches Python-level unhandled exceptions and enables faulthandler for
    fatal interpreter faults. It cannot run for hard process termination.
    """
    global _REPORT_HANDLE, _REPORT_PATH, _INSTALLED
    report_path = (path or default_daemon_report_path()).expanduser().resolve()
    with _LOCK:
        if _INSTALLED:
            return report_path
        report_path.parent.mkdir(parents=True, exist_ok=True)
        fp = report_path.open("a+", encoding="utf-8", buffering=1)
        _REPORT_PATH = report_path
        _REPORT_HANDLE = fp
        if fp.tell() > 0:
            fp.write("\n" + "=" * 80 + "\n")
        fp.write(
            _render_report(
                event="daemon_starting",
                reason="daemon process started",
                details={"report_path": str(report_path)},
            )
        )
        fp.flush()

        try:
            import faulthandler

            faulthandler.enable(file=fp, all_threads=True)
        except Exception as exc:
            fp.write(f"Failed to enable faulthandler: {exc}\n")
            fp.flush()

        previous_excepthook = sys.excepthook

        def _excepthook(exc_type: Any, exc: BaseException, tb: Any) -> None:
            import traceback

            try:
                write_daemon_report(
                    event="daemon_unhandled_exception",
                    reason=str(exc or "unhandled_exception"),
                    details={"exception_type": getattr(exc_type, "__name__", str(exc_type))},
                )
                traceback.print_exception(exc_type, exc, tb, file=fp)
                fp.flush()
            except Exception:
                pass
            previous_excepthook(exc_type, exc, tb)

        sys.excepthook = _excepthook

        previous_threading_excepthook = getattr(threading, "excepthook", None)

        def _threading_excepthook(args: Any) -> None:
            import traceback

            try:
                thread = getattr(args, "thread", None)
                write_daemon_report(
                    event="daemon_thread_unhandled_exception",
                    reason=str(getattr(args, "exc_value", None) or "thread_unhandled_exception"),
                    details={
                        "exception_type": getattr(getattr(args, "exc_type", None), "__name__", str(getattr(args, "exc_type", ""))),
                        "thread_name": str(getattr(thread, "name", "") or ""),
                    },
                )
                traceback.print_exception(args.exc_type, args.exc_value, args.exc_traceback, file=fp)
                fp.flush()
            except Exception:
                pass
            if previous_threading_excepthook is not None:
                previous_threading_excepthook(args)

        if previous_threading_excepthook is not None:
            threading.excepthook = _threading_excepthook

        _INSTALLED = True
        return report_path
