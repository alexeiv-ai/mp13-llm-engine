"""Hosted toolbox callback relay and approval helpers."""
from __future__ import annotations

import asyncio
import concurrent.futures
import inspect
import multiprocessing.connection as mp_connection
import os
import secrets
import tempfile
import threading
from dataclasses import dataclass, field
from multiprocessing.connection import Listener
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from mp13_engine.mp13_toolbox import ToolsView

from .tools_view import (
    _HOSTED_TOOL_APPROVAL_CALLBACK,
    _HOSTED_TOOL_APPROVAL_DECISIONS,
    _approval_timeout_seconds,
    serialize_tools_view,
)
from .windows_ipc import _create_windows_low_integrity_pipe


@dataclass
class HostedToolCallbackContext:
    toolbox_id: str
    tool_name: str
    tool_call_id: Optional[str] = None
    tool_arguments: Dict[str, Any] = field(default_factory=dict)
    engine_id: Optional[str] = None
    callback_name: str = ""
    callback_payload: Any = None
    callback_signature: Optional[Dict[str, Any]] = None
    user_context: Any = None
    gate_outcome: Optional[str] = None
    gate_reason: Optional[str] = None
    requires_confirmation: bool = False

def _request_hosted_tool_approval_with_timeout(
    *,
    processor: Optional[Callable[..., Any]],
    toolbox_id: str,
    tool_name: str,
    tool_call_id: str,
    tool_arguments: Optional[Dict[str, Any]] = None,
    callback_context: Any = None,
    gate_payload: Optional[Dict[str, Any]] = None,
    tools_view: Optional[ToolsView] = None,
    timeout_seconds: Optional[float] = None,
) -> Any:
    if not callable(processor):
        return {"decision": "deny"}
    timeout_value = float(timeout_seconds or _approval_timeout_seconds(callback_context))
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(
            _request_hosted_tool_approval,
            processor=processor,
            toolbox_id=toolbox_id,
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            tool_arguments=tool_arguments,
            callback_context=callback_context,
            gate_payload=gate_payload,
            tools_view=tools_view,
        )
        try:
            return future.result(timeout=timeout_value)
        except concurrent.futures.TimeoutError:
            return {"decision": "deny", "error": "approval_timeout"}
        except Exception:
            return {"decision": "deny"}


class _HostedToolCallbackRelay:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sessions: Dict[str, Dict[str, Any]] = {}
        self._listener: Optional[Listener] = None
        self._listener_family: Optional[str] = None
        self._listener_address: Any = None
        self._listener_thread: Optional[threading.Thread] = None
        self._closed = False

    def _ensure_listener(self) -> None:
        with self._lock:
            if self._listener is not None:
                return
            if os.name == "nt":
                family = "AF_PIPE"
                address = rf"\\.\pipe\mp13-toolbox-callback-{os.getpid()}-{secrets.token_hex(8)}"
            else:
                family = "AF_UNIX"
                callback_root = Path(tempfile.mkdtemp(prefix="mp13-toolbox-callback-")).resolve()
                address = str(callback_root / "callback.sock")
            if os.name == "nt" and family == "AF_PIPE":
                original_new_handle = mp_connection.PipeListener._new_handle

                def _low_integrity_new_handle(pipe_listener: Any, first: bool = False) -> Any:
                    return _create_windows_low_integrity_pipe(str(pipe_listener._address or ""), first=bool(first))

                mp_connection.PipeListener._new_handle = _low_integrity_new_handle
                try:
                    self._listener = Listener(address=address, family=family)
                finally:
                    mp_connection.PipeListener._new_handle = original_new_handle
                pipe_listener = getattr(self._listener, "_listener", None)
                if pipe_listener is not None:
                    setattr(
                        pipe_listener,
                        "_new_handle",
                        lambda first=False, _address=str(getattr(pipe_listener, "_address", "") or ""): _create_windows_low_integrity_pipe(
                            _address,
                            first=bool(first),
                        ),
                    )
            else:
                self._listener = Listener(address=address, family=family)
            self._listener_family = family
            self._listener_address = self._listener.address
            self._listener_thread = threading.Thread(target=self._accept_loop, name="mp13-toolbox-callback-relay", daemon=True)
            self._listener_thread.start()

    def bind_session(
        self,
        *,
        processor: Callable[..., Any],
        toolbox_id: str,
        tool_name: str,
        tool_call_id: str,
        tool_arguments: Optional[Dict[str, Any]] = None,
        callback_signature: Optional[Dict[str, Any]] = None,
        user_context: Any = None,
    ) -> Dict[str, Any]:
        self._ensure_listener()
        session_token = secrets.token_hex(16)
        with self._lock:
            self._sessions[session_token] = {
                "processor": processor,
                "toolbox_id": str(toolbox_id or "").strip(),
                "tool_name": str(tool_name or "").strip(),
                "tool_call_id": str(tool_call_id or "").strip() or None,
                "tool_arguments": dict(tool_arguments or {}),
                "callback_signature": dict(callback_signature or {}) or None,
                "user_context": user_context,
            }
            family = str(self._listener_family or ("AF_PIPE" if os.name == "nt" else "AF_UNIX"))
            address = self._listener_address
        return {
            "family": family,
            "address": address,
            "session_token": session_token,
            "contract": "hosting.toolbox.callbacks.v2",
        }

    def release_session(self, session_token: str) -> None:
        token = str(session_token or "").strip()
        if not token:
            return
        with self._lock:
            self._sessions.pop(token, None)

    def _accept_loop(self) -> None:
        listener = self._listener
        if listener is None:
            return
        while not self._closed:
            try:
                conn = listener.accept()
            except (OSError, EOFError):
                return
            thread = threading.Thread(target=self._handle_connection, args=(conn,), daemon=True)
            thread.start()

    @staticmethod
    def _invoke_processor(processor: Callable[..., Any], *, callback_name: str, payload: Any, context: HostedToolCallbackContext) -> Any:
        try:
            result = processor(callback_name=callback_name, payload=payload, context=context)
        except TypeError:
            result = processor(callback_name, payload, context)
        if inspect.isawaitable(result):
            return asyncio.run(result)
        return result

    def _handle_connection(self, conn: Any) -> None:
        try:
            payload = dict(conn.recv() or {})
            session_token = str(payload.get("session_token") or "").strip()
            callback_name = str(payload.get("callback_name") or "").strip()
            callback_payload = payload.get("payload")
            callback_context = dict(payload.get("context") or {})
            with self._lock:
                session = dict(self._sessions.get(session_token) or {})
            if not session:
                conn.send({"status": "error", "message": "callback_session_missing"})
                return
            processor = session.get("processor")
            if not callable(processor):
                conn.send({"status": "error", "message": "callback_processor_missing"})
                return
            context = HostedToolCallbackContext(
                toolbox_id=str(session.get("toolbox_id") or "").strip(),
                tool_name=str(session.get("tool_name") or "").strip(),
                tool_call_id=str(session.get("tool_call_id") or "").strip() or None,
                tool_arguments=dict(session.get("tool_arguments") or {}),
                engine_id=str(callback_context.get("engine_id") or "").strip() or None,
                callback_name=callback_name,
                callback_payload=callback_payload,
                callback_signature=dict(session.get("callback_signature") or {}) or None,
                user_context=session.get("user_context"),
            )
            result = self._invoke_processor(
                processor,
                callback_name=callback_name,
                payload=callback_payload,
                context=context,
            )
            conn.send({"status": "ok", "result": result})
        except Exception as exc:
            try:
                conn.send({"status": "error", "message": f"callback_processor_failed:{type(exc).__name__}:{exc}"})
            except Exception:
                pass
        finally:
            try:
                conn.close()
            except Exception:
                pass


def _request_hosted_tool_approval(
    *,
    processor: Optional[Callable[..., Any]],
    toolbox_id: str,
    tool_name: str,
    tool_call_id: str,
    tool_arguments: Optional[Dict[str, Any]] = None,
    callback_context: Any = None,
    gate_payload: Optional[Dict[str, Any]] = None,
    tools_view: Optional[ToolsView] = None,
) -> str:
    if not callable(processor):
        return "deny"
    gate = dict(gate_payload or {})
    context = HostedToolCallbackContext(
        toolbox_id=str(toolbox_id or "").strip(),
        tool_name=str(tool_name or "").strip(),
        tool_call_id=str(tool_call_id or "").strip() or None,
        tool_arguments=dict(tool_arguments or {}),
        callback_name=_HOSTED_TOOL_APPROVAL_CALLBACK,
        callback_payload={
            "kind": "tool_approval_request",
            "decision_options": list(_HOSTED_TOOL_APPROVAL_DECISIONS),
            "tool_name": str(tool_name or "").strip(),
            "tool_call_id": str(tool_call_id or "").strip() or None,
            "tool_arguments": dict(tool_arguments or {}),
            "gate": {
                "outcome": str(gate.get("outcome") or "").strip() or "gated_requires_confirmation",
                "reason": str(gate.get("reason") or "").strip() or "gated_requires_confirmation",
                "requires_confirmation": bool(gate.get("requires_confirmation", True)),
            },
            "tools_view": serialize_tools_view(tools_view),
        },
        user_context=callback_context,
        gate_outcome=str(gate.get("outcome") or "").strip() or "gated_requires_confirmation",
        gate_reason=str(gate.get("reason") or "").strip() or "gated_requires_confirmation",
        requires_confirmation=bool(gate.get("requires_confirmation", True)),
    )
    result = _HostedToolCallbackRelay._invoke_processor(
        processor,
        callback_name=_HOSTED_TOOL_APPROVAL_CALLBACK,
        payload=context.callback_payload,
        context=context,
    )
    return result
