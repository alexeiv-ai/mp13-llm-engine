from __future__ import annotations

import asyncio
import hashlib
import importlib
import inspect
import json
import os
import secrets
import socket
import subprocess
import sys
import tempfile
import threading
import time
import venv
import ctypes
import multiprocessing.connection as mp_connection
from ctypes import wintypes
from dataclasses import dataclass, field
from multiprocessing.connection import Client, Listener
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from mp13_engine.mp13_config import InferenceResponse, ParserProfile, ToolCall, ToolCallBlock
from mp13_engine.mp13_toolbox import Toolbox, ToolsView
from mp13_engine.mp13_tools_parser import UnifiedToolIO


if os.name == "nt":
    _advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
    _SDDL_REVISION_1 = 1
    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)


    class _SECURITY_ATTRIBUTES(ctypes.Structure):
        _fields_ = [
            ("nLength", wintypes.DWORD),
            ("lpSecurityDescriptor", wintypes.LPVOID),
            ("bInheritHandle", wintypes.BOOL),
        ]

    _advapi32.ConvertStringSecurityDescriptorToSecurityDescriptorW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.LPVOID),
        ctypes.POINTER(wintypes.DWORD),
    ]
    _advapi32.ConvertStringSecurityDescriptorToSecurityDescriptorW.restype = wintypes.BOOL
    _kernel32.CreateNamedPipeW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.POINTER(_SECURITY_ATTRIBUTES),
    ]
    _kernel32.CreateNamedPipeW.restype = wintypes.HANDLE
    _kernel32.LocalFree.argtypes = [wintypes.HLOCAL]
    _kernel32.LocalFree.restype = wintypes.HLOCAL


def _create_windows_low_integrity_pipe(address: str, *, first: bool = False) -> Any:
    if os.name != "nt":
        raise RuntimeError("windows_pipe_only")
    pipe_name = str(address or "").strip()
    if not pipe_name:
        raise ValueError("pipe_name_required")
    sd = wintypes.LPVOID()
    sd_size = wintypes.DWORD(0)
    if not _advapi32.ConvertStringSecurityDescriptorToSecurityDescriptorW(
        "S:(ML;;NW;;;LW)",
        _SDDL_REVISION_1,
        ctypes.byref(sd),
        ctypes.byref(sd_size),
    ):
        err = ctypes.get_last_error()
        raise OSError(err, f"ConvertStringSecurityDescriptorToSecurityDescriptorW failed for {pipe_name}")
    try:
        security = _SECURITY_ATTRIBUTES(
            nLength=ctypes.sizeof(_SECURITY_ATTRIBUTES),
            lpSecurityDescriptor=sd,
            bInheritHandle=False,
        )
        flags = mp_connection._winapi.PIPE_ACCESS_DUPLEX | mp_connection._winapi.FILE_FLAG_OVERLAPPED
        if first:
            flags |= mp_connection._winapi.FILE_FLAG_FIRST_PIPE_INSTANCE
        handle = _kernel32.CreateNamedPipeW(
            ctypes.c_wchar_p(pipe_name),
            flags,
            mp_connection._winapi.PIPE_TYPE_MESSAGE
            | mp_connection._winapi.PIPE_READMODE_MESSAGE
            | mp_connection._winapi.PIPE_WAIT,
            mp_connection._winapi.PIPE_UNLIMITED_INSTANCES,
            mp_connection.BUFSIZE,
            mp_connection.BUFSIZE,
            mp_connection._winapi.NMPWAIT_WAIT_FOREVER,
            ctypes.byref(security),
        )
        if not handle or int(handle) == -1:
            err = ctypes.get_last_error()
            raise OSError(err, f"CreateNamedPipeW failed for {pipe_name}")
        return int(handle)
    finally:
        _kernel32.LocalFree(sd)


def _stable_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def serialize_tools_view(tools_view: Optional[ToolsView]) -> Optional[Dict[str, Any]]:
    if tools_view is None:
        return None
    return {
        "view_id": str(tools_view.view_id or "").strip(),
        "mode": str(tools_view.mode or "").strip(),
        "allowed_tools": sorted(str(item or "").strip() for item in list(tools_view.allowed_tools or []) if str(item or "").strip()),
        "advertised_tools": sorted(str(item or "").strip() for item in list(tools_view.advertised_tools or []) if str(item or "").strip()),
        "hidden_allowed_tools": sorted(str(item or "").strip() for item in list(tools_view.hidden_allowed_tools or []) if str(item or "").strip()),
        "disabled_tools": sorted(str(item or "").strip() for item in list(tools_view.disabled_tools or []) if str(item or "").strip()),
        "gated_tools": sorted(str(item or "").strip() for item in list(tools_view.gated_tools or []) if str(item or "").strip()),
    }


def is_canceled_tool_error(tool_call: Any) -> bool:
    if isinstance(tool_call, dict):
        error_text = str(tool_call.get("error") or "").strip().lower()
    else:
        error_text = str(getattr(tool_call, "error", "") or "").strip().lower()
    return error_text == "canceled" or error_text.startswith("execution canceled:")


def should_resubmit_canceled_tool_call(
    tool_call: Any,
    *,
    non_restartable: bool = False,
) -> bool:
    return is_canceled_tool_error(tool_call) and not bool(non_restartable)


def _is_coarse_cancel_execution_error(exc: BaseException) -> bool:
    message = str(exc or "").strip().lower()
    if not message:
        return False
    cancel_markers = (
        "toolbox_executor_missing",
        "engine_not_found",
        "no output",
        "connection reset",
        "broken pipe",
        "end of file",
        "eoferror",
        "worker_exception",
    )
    return any(marker in message for marker in cancel_markers)


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
            "user_context": user_context,
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


@dataclass
class ToolboxBundleFile:
    relative_path: str
    content: str

    def normalized_path(self) -> str:
        raw = str(self.relative_path or "").replace("\\", "/").strip("/")
        if not raw or raw.startswith("../") or "/../" in f"/{raw}/":
            raise ValueError("bundle_file_path_invalid")
        return raw

    def to_dict(self) -> Dict[str, Any]:
        return {
            "relative_path": self.normalized_path(),
            "content_sha256": _sha256_text(str(self.content or "")),
        }

    def to_runtime_dict(self) -> Dict[str, Any]:
        return {
            "relative_path": self.normalized_path(),
            "content": str(self.content or ""),
        }

    @classmethod
    def from_runtime_dict(cls, payload: Dict[str, Any]) -> "ToolboxBundleFile":
        row = dict(payload or {})
        return cls(
            relative_path=str(row.get("relative_path") or "").strip(),
            content=str(row.get("content") or ""),
        )


@dataclass
class ToolboxBundleTool:
    definition: Dict[str, Any]
    entrypoint: str
    hidden: bool = False
    non_restartable: bool = False
    callback_signature: Optional[Dict[str, Any]] = None

    def tool_name(self) -> str:
        fn = dict(self.definition.get("function") or {})
        name = str(fn.get("name") or "").strip()
        if not name:
            raise ValueError("tool_name_required")
        return name

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.tool_name(),
            "definition": dict(self.definition or {}),
            "entrypoint": str(self.entrypoint or "").strip(),
            "hidden": bool(self.hidden),
            "non_restartable": bool(self.non_restartable),
            "callback_signature": dict(self.callback_signature or {}) or None,
        }


@dataclass
class ToolboxBundleAutoTool:
    module_name: str
    callable_name: str
    activate: bool = True
    hidden: bool = False
    non_restartable: bool = False
    guide_content: Optional[Dict[str, List[str]]] = None
    guide_description: Optional[str] = None
    callback_signature: Optional[Dict[str, Any]] = None

    def normalized_module_name(self) -> str:
        raw = str(self.module_name or "").strip()
        if not raw:
            raise ValueError("auto_tool_module_name_required")
        return raw

    def normalized_callable_name(self) -> str:
        raw = str(self.callable_name or "").strip()
        if not raw:
            raise ValueError("auto_tool_callable_name_required")
        return raw

    def tool_name(self) -> str:
        return self.normalized_callable_name()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.tool_name(),
            "module_name": self.normalized_module_name(),
            "callable_name": self.normalized_callable_name(),
            "activate": bool(self.activate),
            "hidden": bool(self.hidden),
            "non_restartable": bool(self.non_restartable),
            "guide_content": dict(self.guide_content or {}) or None,
            "guide_description": str(self.guide_description or "").strip() or None,
            "callback_signature": dict(self.callback_signature or {}) or None,
        }


@dataclass
class SandboxProfileSpec:
    profile_id: str = ""
    environment_name: str = ""
    required_imports: List[str] = field(default_factory=list)
    sandbox_policy: Dict[str, Any] = field(default_factory=dict)

    def normalized_profile_id(self) -> str:
        raw = str(self.profile_id or "").strip()
        if raw:
            return raw
        return f"profile-{self.profile_fingerprint()[:12]}"

    def normalized_required_imports(self) -> List[str]:
        imports: List[str] = []
        seen: set[str] = set()
        for item in list(self.required_imports or []):
            name = str(item or "").strip()
            if name and name not in seen:
                seen.add(name)
                imports.append(name)
        return imports

    def profile_fingerprint(self) -> str:
        payload = {
            "environment_name": str(self.environment_name or "").strip() or "base",
            "required_imports": self.normalized_required_imports(),
            "sandbox_policy": dict(self.sandbox_policy or {}),
        }
        return _sha256_text(_stable_json(payload))

    def intrinsics_profile_id(self, intrinsic_tool_names: Sequence[Any]) -> str:
        names = {
            str(item or "").strip()
            for item in list(intrinsic_tool_names or [])
            if str(item or "").strip()
        }
        uses_calculator = bool(
            {"scriptable_calculator", "scriptable_calculator_guide"} & names
        )
        uses_symbolic = bool(
            {"symbolic_algebra", "symbolic_algebra_guide"} & names
        )
        if uses_calculator and uses_symbolic:
            return "calculator+symbolic_math"
        if uses_symbolic:
            return "symbolic_math"
        if uses_calculator:
            return "calculator"
        return "none"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "profile_id": self.normalized_profile_id(),
            "environment_name": str(self.environment_name or "").strip() or "base",
            "required_imports": self.normalized_required_imports(),
            "sandbox_policy": dict(self.sandbox_policy or {}),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "SandboxProfileSpec":
        row = dict(payload or {})
        return cls(
            profile_id=str(row.get("profile_id") or "").strip(),
            environment_name=str(row.get("environment_name") or "base").strip() or "base",
            required_imports=[str(item or "").strip() for item in list(row.get("required_imports") or []) if str(item or "").strip()],
            sandbox_policy=dict(row.get("sandbox_policy") or {}),
        )


@dataclass
class ToolboxAutoAssignmentRequest:
    files: List[ToolboxBundleFile]
    module_name: str
    callable_name: str
    sandbox_profile: SandboxProfileSpec = field(default_factory=SandboxProfileSpec)
    activate: bool = True
    hidden: bool = False
    non_restartable: bool = False
    guide_content: Optional[Dict[str, List[str]]] = None
    guide_description: Optional[str] = None
    callback_signature: Optional[Dict[str, Any]] = None

    def to_auto_tool(self) -> ToolboxBundleAutoTool:
        return ToolboxBundleAutoTool(
            module_name=str(self.module_name or "").strip(),
            callable_name=str(self.callable_name or "").strip(),
            activate=bool(self.activate),
            hidden=bool(self.hidden),
            non_restartable=bool(self.non_restartable),
            guide_content=dict(self.guide_content or {}) or None,
            guide_description=str(self.guide_description or "").strip() or None,
            callback_signature=dict(self.callback_signature or {}) or None,
        )

    def stable_key(self) -> str:
        return f"{str(self.module_name or '').strip()}:{str(self.callable_name or '').strip()}"

    def to_runtime_dict(self) -> Dict[str, Any]:
        return {
            "files": [item.to_runtime_dict() for item in list(self.files or [])],
            "module_name": str(self.module_name or "").strip(),
            "callable_name": str(self.callable_name or "").strip(),
            "sandbox_profile": self.sandbox_profile.to_dict(),
            "activate": bool(self.activate),
            "hidden": bool(self.hidden),
            "non_restartable": bool(self.non_restartable),
            "guide_content": dict(self.guide_content or {}) or None,
            "guide_description": str(self.guide_description or "").strip() or None,
            "callback_signature": dict(self.callback_signature or {}) or None,
        }

    @classmethod
    def from_runtime_dict(cls, payload: Dict[str, Any]) -> "ToolboxAutoAssignmentRequest":
        row = dict(payload or {})
        return cls(
            files=[ToolboxBundleFile.from_runtime_dict(dict(item or {})) for item in list(row.get("files") or [])],
            module_name=str(row.get("module_name") or "").strip(),
            callable_name=str(row.get("callable_name") or "").strip(),
            sandbox_profile=SandboxProfileSpec.from_dict(dict(row.get("sandbox_profile") or {})),
            activate=bool(row.get("activate", True)),
            hidden=bool(row.get("hidden", False)),
            non_restartable=bool(row.get("non_restartable", False)),
            guide_content=dict(row.get("guide_content") or {}) or None,
            guide_description=str(row.get("guide_description") or "").strip() or None,
            callback_signature=dict(row.get("callback_signature") or {}) or None,
        )


@dataclass
class ToolboxManualAssignmentRequest:
    files: List[ToolboxBundleFile]
    module_name: str
    callable_name: str
    tool_definition: Dict[str, Any]
    sandbox_profile: SandboxProfileSpec = field(default_factory=SandboxProfileSpec)
    hidden: bool = False
    non_restartable: bool = False
    callback_signature: Optional[Dict[str, Any]] = None

    def to_bundle_tool(self) -> ToolboxBundleTool:
        return ToolboxBundleTool(
            definition=dict(self.tool_definition or {}),
            entrypoint=f"{str(self.module_name or '').strip()}:{str(self.callable_name or '').strip()}",
            hidden=bool(self.hidden),
            non_restartable=bool(self.non_restartable),
            callback_signature=dict(self.callback_signature or {}) or None,
        )

    def stable_key(self) -> str:
        return f"manual:{str(self.module_name or '').strip()}:{str(self.callable_name or '').strip()}"

    def to_runtime_dict(self) -> Dict[str, Any]:
        return {
            "files": [item.to_runtime_dict() for item in list(self.files or [])],
            "module_name": str(self.module_name or "").strip(),
            "callable_name": str(self.callable_name or "").strip(),
            "tool_definition": dict(self.tool_definition or {}),
            "sandbox_profile": self.sandbox_profile.to_dict(),
            "hidden": bool(self.hidden),
            "non_restartable": bool(self.non_restartable),
            "callback_signature": dict(self.callback_signature or {}) or None,
        }

    @classmethod
    def from_runtime_dict(cls, payload: Dict[str, Any]) -> "ToolboxManualAssignmentRequest":
        row = dict(payload or {})
        return cls(
            files=[ToolboxBundleFile.from_runtime_dict(dict(item or {})) for item in list(row.get("files") or [])],
            module_name=str(row.get("module_name") or "").strip(),
            callable_name=str(row.get("callable_name") or "").strip(),
            tool_definition=dict(row.get("tool_definition") or {}),
            sandbox_profile=SandboxProfileSpec.from_dict(dict(row.get("sandbox_profile") or {})),
            hidden=bool(row.get("hidden", False)),
            non_restartable=bool(row.get("non_restartable", False)),
            callback_signature=dict(row.get("callback_signature") or {}) or None,
        )


@dataclass
class ToolboxSandboxAssignment:
    toolbox_id: str
    sandbox_profile: SandboxProfileSpec
    bundle_spec: "ToolboxBundleSpec"
    staged_bundle: Optional["StagedToolboxBundle"] = None
    registration: Optional[Dict[str, Any]] = None


@dataclass
class ToolboxBundleSpec:
    bundle_id: str
    toolbox_id: Optional[str] = None
    sandbox_profile: Optional[SandboxProfileSpec] = None
    files: List[ToolboxBundleFile] = field(default_factory=list)
    tools: List[ToolboxBundleTool] = field(default_factory=list)
    auto_tools: List[ToolboxBundleAutoTool] = field(default_factory=list)
    with_intrinsics: bool = False
    with_intrinsic_guides: bool = False
    intrinsic_tool_names: List[str] = field(default_factory=list)
    active_intrinsic_tool_names: List[str] = field(default_factory=list)
    hidden_intrinsic_tool_names: List[str] = field(default_factory=list)
    hidden_tool_names: List[str] = field(default_factory=list)
    dependency_lock_hash: Optional[str] = None

    def normalized_bundle_id(self) -> str:
        raw = str(self.bundle_id or "").strip()
        if not raw:
            raise ValueError("bundle_id_required")
        return raw

    def normalized_toolbox_id(self) -> str:
        raw = str(self.toolbox_id or "").strip()
        return raw or self.normalized_bundle_id()

    def normalized_intrinsic_tool_names(self) -> List[str]:
        out: List[str] = []
        seen: set[str] = set()
        for item in list(self.intrinsic_tool_names or []):
            name = str(item or "").strip()
            if name and name not in seen:
                seen.add(name)
                out.append(name)
        return out

    @staticmethod
    def _normalize_name_list(items: Sequence[Any]) -> List[str]:
        out: List[str] = []
        seen: set[str] = set()
        for item in list(items or []):
            name = str(item or "").strip()
            if name and name not in seen:
                seen.add(name)
                out.append(name)
        return out

    def manifest_payload(self) -> Dict[str, Any]:
        bundle_id = self.normalized_bundle_id()
        toolbox_id = self.normalized_toolbox_id()
        sandbox_profile = (self.sandbox_profile or SandboxProfileSpec(profile_id="default")).to_dict()
        tools = [item.to_dict() for item in self.tools]
        auto_tools = [item.to_dict() for item in self.auto_tools]
        intrinsic_tool_names = self.normalized_intrinsic_tool_names()
        if not tools and not auto_tools and not intrinsic_tool_names:
            raise ValueError("bundle_tools_required")
        files = [item.to_dict() for item in self.files]
        active_intrinsic_tool_names = self._normalize_name_list(
            self.active_intrinsic_tool_names if self.active_intrinsic_tool_names else intrinsic_tool_names
        )
        hidden_intrinsic_tool_names = self._normalize_name_list(self.hidden_intrinsic_tool_names)
        hidden_tool_names = self._normalize_name_list(
            list(self.hidden_tool_names)
            + [item.tool_name() for item in list(self.tools or []) if bool(getattr(item, "hidden", False))]
            + [item.tool_name() for item in list(self.auto_tools or []) if bool(getattr(item, "hidden", False))]
        )
        manifest_input = {
            "bundle_id": bundle_id,
            "toolbox_id": toolbox_id,
            "sandbox_profile": sandbox_profile,
            "tools": tools,
            "auto_tools": auto_tools,
            "files": files,
            "with_intrinsics": bool(self.with_intrinsics or bool(intrinsic_tool_names)),
            "with_intrinsic_guides": bool(self.with_intrinsic_guides),
            "intrinsic_tool_names": intrinsic_tool_names,
            "active_intrinsic_tool_names": active_intrinsic_tool_names,
            "hidden_intrinsic_tool_names": hidden_intrinsic_tool_names,
            "hidden_tool_names": hidden_tool_names,
            "dependency_lock_hash": str(self.dependency_lock_hash or "").strip() or None,
        }
        manifest_hash = _sha256_text(_stable_json(manifest_input))
        bundle_revision = manifest_hash[:16]
        return {
            "executor_kind": "toolbox_executor_v1",
            "bundle_id": bundle_id,
            "toolbox_id": toolbox_id,
            "sandbox_profile": sandbox_profile,
            "bundle_revision": bundle_revision,
            "manifest_hash": manifest_hash,
            "dependency_lock_hash": str(self.dependency_lock_hash or "").strip() or None,
            "tools": tools,
            "auto_tools": auto_tools,
            "files": files,
            "with_intrinsics": bool(self.with_intrinsics or bool(intrinsic_tool_names)),
            "with_intrinsic_guides": bool(self.with_intrinsic_guides),
            "intrinsic_tool_names": intrinsic_tool_names,
            "active_intrinsic_tool_names": active_intrinsic_tool_names,
            "hidden_intrinsic_tool_names": hidden_intrinsic_tool_names,
            "hidden_tool_names": hidden_tool_names,
        }


@dataclass
class ToolboxWorkerStartupSpec:
    worker_id: str
    sandbox_id: str
    toolbox_revision: str
    manifest_path: str
    scratch_root: str
    engines_state_file: Optional[str] = None
    control_state_file: Optional[str] = None
    venv_path: Optional[str] = None
    ipc_family: str = field(default_factory=lambda: "AF_PIPE" if os.name == "nt" else "AF_UNIX")
    ipc_address: str = ""
    auth_token_env: str = "MP13_ENGINE_HOST_TOKEN"
    execution_contract: str = "hosting.toolbox.worker.v1"
    callback_contract: str = "hosting.toolbox.callbacks.v1"
    policy: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        default_ipc_family = "AF_PIPE" if os.name == "nt" else "AF_UNIX"
        return {
            "worker_id": str(self.worker_id or "").strip(),
            "sandbox_id": str(self.sandbox_id or "").strip(),
            "toolbox_revision": str(self.toolbox_revision or "").strip(),
            "manifest_path": str(self.manifest_path or "").strip(),
            "scratch_root": str(self.scratch_root or "").strip(),
            "engines_state_file": str(self.engines_state_file or "").strip() or None,
            "control_state_file": str(self.control_state_file or "").strip() or None,
            "venv_path": str(self.venv_path or "").strip() or None,
            "ipc_family": str(self.ipc_family or default_ipc_family).strip() or default_ipc_family,
            "ipc_address": str(self.ipc_address or "").strip(),
            "auth_token_env": str(self.auth_token_env or "MP13_ENGINE_HOST_TOKEN").strip() or "MP13_ENGINE_HOST_TOKEN",
            "execution_contract": str(self.execution_contract or "hosting.toolbox.worker.v1").strip() or "hosting.toolbox.worker.v1",
            "callback_contract": str(self.callback_contract or "hosting.toolbox.callbacks.v1").strip() or "hosting.toolbox.callbacks.v1",
            "policy": dict(self.policy or {}),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ToolboxWorkerStartupSpec":
        row = dict(payload or {})
        default_ipc_family = "AF_PIPE" if os.name == "nt" else "AF_UNIX"
        return cls(
            worker_id=str(row.get("worker_id") or "").strip(),
            sandbox_id=str(row.get("sandbox_id") or "").strip(),
            toolbox_revision=str(row.get("toolbox_revision") or "").strip(),
            manifest_path=str(row.get("manifest_path") or "").strip(),
            scratch_root=str(row.get("scratch_root") or "").strip(),
            engines_state_file=str(row.get("engines_state_file") or "").strip() or None,
            control_state_file=str(row.get("control_state_file") or "").strip() or None,
            venv_path=str(row.get("venv_path") or "").strip() or None,
            ipc_family=str(row.get("ipc_family") or default_ipc_family).strip() or default_ipc_family,
            ipc_address=str(row.get("ipc_address") or "").strip(),
            auth_token_env=str(row.get("auth_token_env") or "MP13_ENGINE_HOST_TOKEN").strip() or "MP13_ENGINE_HOST_TOKEN",
            execution_contract=str(row.get("execution_contract") or "hosting.toolbox.worker.v1").strip() or "hosting.toolbox.worker.v1",
            callback_contract=str(row.get("callback_contract") or "hosting.toolbox.callbacks.v1").strip() or "hosting.toolbox.callbacks.v1",
            policy=dict(row.get("policy") or {}),
        )

    def write_json(self, path: Path) -> Path:
        target = Path(path).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return target


@dataclass
class ToolboxEnvironmentSpec:
    venv_key: str
    venv_path: str
    python_executable: str = ""
    environment_name: str = "base"
    environment_description_hash: str = ""
    venv_lock_hash: Optional[str] = None
    toolbox_runtime_hash: str = "toolbox-executor-v1"
    intrinsics_profile_id: str = "none"
    required_imports: List[str] = field(default_factory=list)
    dependency_lock_hash: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "venv_key": str(self.venv_key or "").strip(),
            "venv_path": str(self.venv_path or "").strip(),
            "python_executable": str(self.python_executable or "").strip(),
            "environment_name": str(self.environment_name or "base").strip() or "base",
            "environment_description_hash": str(self.environment_description_hash or "").strip() or None,
            "venv_lock_hash": str(self.venv_lock_hash or "").strip() or None,
            "toolbox_runtime_hash": str(self.toolbox_runtime_hash or "toolbox-executor-v1").strip() or "toolbox-executor-v1",
            "intrinsics_profile_id": str(self.intrinsics_profile_id or "none").strip() or "none",
            "required_imports": [str(item or "").strip() for item in list(self.required_imports or []) if str(item or "").strip()],
            "dependency_lock_hash": str(self.dependency_lock_hash or "").strip() or None,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ToolboxEnvironmentSpec":
        row = dict(payload or {})
        return cls(
            venv_key=str(row.get("venv_key") or "").strip(),
            venv_path=str(row.get("venv_path") or "").strip(),
            python_executable=str(row.get("python_executable") or "").strip(),
            environment_name=str(row.get("environment_name") or "base").strip() or "base",
            environment_description_hash=str(row.get("environment_description_hash") or "").strip() or None,
            venv_lock_hash=str(row.get("venv_lock_hash") or "").strip() or None,
            toolbox_runtime_hash=str(row.get("toolbox_runtime_hash") or "toolbox-executor-v1").strip() or "toolbox-executor-v1",
            intrinsics_profile_id=str(row.get("intrinsics_profile_id") or "none").strip() or "none",
            required_imports=[str(item or "").strip() for item in list(row.get("required_imports") or []) if str(item or "").strip()],
            dependency_lock_hash=str(row.get("dependency_lock_hash") or "").strip() or None,
        )


class ToolboxEnvironmentManager:
    def __init__(self, hosting_root: Path):
        self.hosting_root = Path(hosting_root).expanduser().resolve()
        self.environments_root = (self.hosting_root / "toolbox_venvs").resolve()

    @staticmethod
    def normalize_environment_description(
        payload: Optional[Dict[str, Any]],
        *,
        name: Optional[str] = None,
    ) -> Dict[str, Any]:
        row = dict(payload or {})
        env_name = str(name or row.get("name") or "base").strip() or "base"
        base_env_name = str(row.get("base_env_name") or ("base" if env_name != "base" else "")).strip()
        extra_packages: List[str] = []
        seen: set[str] = set()
        for item in list(row.get("extra_packages") or []):
            pkg = str(item or "").strip()
            if pkg and pkg not in seen:
                seen.add(pkg)
                extra_packages.append(pkg)
        return {
            "name": env_name,
            "base_env_name": base_env_name or None,
            "extra_packages": extra_packages,
            "allow_online_install": bool(row.get("allow_online_install", False)),
        }

    @classmethod
    def environment_description_hash(cls, payload: Optional[Dict[str, Any]], *, name: Optional[str] = None) -> str:
        normalized = cls.normalize_environment_description(payload, name=name)
        return cls._fingerprint_payload(normalized)[:16]

    @classmethod
    def resolve_environment_description(
        cls,
        payload_by_name: Dict[str, Dict[str, Any]],
        *,
        name: str,
    ) -> Dict[str, Any]:
        env_name = str(name or "base").strip() or "base"
        seen_stack: set[str] = set()
        lineage: List[str] = []
        merged_packages: List[str] = []
        merged_seen: set[str] = set()
        allow_online_install = False

        current = env_name
        while current:
            normalized = cls.normalize_environment_description(
                dict(payload_by_name.get(current) or {}),
                name=current,
            )
            if current in seen_stack:
                raise ValueError(f"environment description cycle detected at '{current}'")
            seen_stack.add(current)
            lineage.append(current)
            for item in list(normalized.get("extra_packages") or []):
                pkg = str(item or "").strip()
                if pkg and pkg not in merged_seen:
                    merged_seen.add(pkg)
                    merged_packages.append(pkg)
            allow_online_install = bool(allow_online_install or normalized.get("allow_online_install", False))
            base_env_name = str(normalized.get("base_env_name") or "").strip()
            current = base_env_name if base_env_name and base_env_name != normalized["name"] else ""

        direct = cls.normalize_environment_description(dict(payload_by_name.get(env_name) or {}), name=env_name)
        return {
            "name": env_name,
            "base_env_name": direct.get("base_env_name"),
            "extra_packages": list(direct.get("extra_packages") or []),
            "allow_online_install": bool(direct.get("allow_online_install", False)),
            "effective_extra_packages": merged_packages,
            "effective_allow_online_install": allow_online_install,
            "lineage": lineage,
        }

    @staticmethod
    def _fingerprint_payload(payload: Dict[str, Any]) -> str:
        return _sha256_text(_stable_json(payload))

    def environment_spec_for_bundle(
        self,
        staged: "StagedToolboxBundle",
        *,
        environment_description: Optional[Dict[str, Any]] = None,
    ) -> ToolboxEnvironmentSpec:
        manifest = dict(staged.manifest or {})
        sandbox_profile = SandboxProfileSpec.from_dict(dict(manifest.get("sandbox_profile") or {}))
        intrinsic_tool_names = list(manifest.get("intrinsic_tool_names") or [])
        intrinsics_profile_id = sandbox_profile.intrinsics_profile_id(intrinsic_tool_names)
        dependency_lock_hash = str(manifest.get("dependency_lock_hash") or "").strip() or None
        required_imports = sandbox_profile.normalized_required_imports()
        toolbox_runtime_hash = "toolbox-executor-v1"
        environment_name = str(sandbox_profile.environment_name or "base").strip() or "base"
        input_desc = dict(environment_description or {})
        raw_desc = self.normalize_environment_description(input_desc, name=environment_name)
        effective_extra_packages = [
            str(item or "").strip()
            for item in list(input_desc.get("effective_extra_packages") or raw_desc.get("extra_packages") or [])
            if str(item or "").strip()
        ]
        env_desc = {
            "name": environment_name,
            "base_env_name": raw_desc.get("base_env_name"),
            "extra_packages": effective_extra_packages,
            "allow_online_install": bool(
                input_desc.get("effective_allow_online_install", raw_desc.get("allow_online_install", False))
            ),
        }
        env_desc_hash = self.environment_description_hash(env_desc, name=environment_name)
        venv_key = self._fingerprint_payload(
            {
                "toolbox_runtime_hash": toolbox_runtime_hash,
                "environment_name": environment_name,
                "environment_description_hash": env_desc_hash,
                "intrinsics_profile_id": intrinsics_profile_id,
                "required_imports": required_imports,
                "dependency_lock_hash": dependency_lock_hash,
            }
        )[:16]
        venv_root = (self.environments_root / venv_key).resolve()
        venv_path = str(venv_root)
        venv_lock_hash = dependency_lock_hash or self._fingerprint_payload(
            {
                "intrinsics_profile_id": intrinsics_profile_id,
                "required_imports": required_imports,
            }
        )[:16]
        return ToolboxEnvironmentSpec(
            venv_key=venv_key,
            venv_path=venv_path,
            python_executable=str(self.python_executable_path(venv_root)),
            environment_name=environment_name,
            environment_description_hash=env_desc_hash,
            venv_lock_hash=venv_lock_hash,
            toolbox_runtime_hash=toolbox_runtime_hash,
            intrinsics_profile_id=intrinsics_profile_id,
            required_imports=required_imports,
            dependency_lock_hash=dependency_lock_hash,
        )

    @staticmethod
    def python_executable_path(venv_root: Path) -> Path:
        base = Path(venv_root).expanduser().resolve()
        if os.name == "nt":
            return base / "Scripts" / "python.exe"
        return base / "bin" / "python"

    def ensure_environment(self, spec: ToolboxEnvironmentSpec) -> ToolboxEnvironmentSpec:
        target = Path(spec.venv_path).expanduser().resolve()
        if not (target / "pyvenv.cfg").exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            # Reuse the current interpreter's site packages for now so sandbox worker
            # execution remains functional before locked dependency installs are added.
            venv.EnvBuilder(with_pip=False, system_site_packages=True).create(str(target))
        spec.python_executable = str(self.python_executable_path(target))
        metadata_path = target / "environment.json"
        metadata = self.read_environment_metadata(spec) if metadata_path.exists() else {}
        metadata.update(spec.to_dict())
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        return spec

    def runtime_python_executable(
        self,
        spec: ToolboxEnvironmentSpec,
        *,
        fallback_python_executable: Optional[str] = None,
    ) -> str:
        ensured = self.ensure_environment(spec)
        env_root = Path(ensured.venv_path).expanduser().resolve()
        env_python = str(ensured.python_executable or self.python_executable_path(env_root)).strip()
        fallback_python = str(fallback_python_executable or "").strip()
        if not fallback_python:
            return env_python
        metadata = self.read_environment_metadata(ensured)
        install_execution_status = str(dict(metadata.get("install_execution") or {}).get("status") or "").strip().lower()
        receipt_verification_status = str(
            dict(metadata.get("install_receipt_verification") or {}).get("status") or ""
        ).strip().lower()
        if install_execution_status == "ok" and receipt_verification_status == "ok":
            return env_python
        return fallback_python

    @staticmethod
    def _unique_names(items: Sequence[Any]) -> List[str]:
        out: List[str] = []
        seen: set[str] = set()
        for item in list(items or []):
            name = str(item or "").strip()
            if name and name not in seen:
                seen.add(name)
                out.append(name)
        return out

    @staticmethod
    def _normalize_package_name(value: Any) -> str:
        raw = str(value or "").strip()
        if not raw:
            return ""
        for sep in ("==", ">=", "<=", "~=", "!=", ">", "<"):
            if sep in raw:
                raw = raw.split(sep, 1)[0]
                break
        raw = raw.strip()
        return raw.lower()

    @classmethod
    def _install_plan_hash(cls, install_plan: Dict[str, Any]) -> str:
        payload = {
            "planned_packages": cls._unique_names(install_plan.get("planned_packages") or []),
            "requirements_relpath": str(install_plan.get("requirements_relpath") or "").strip() or "requirements-planned.txt",
        }
        return cls._fingerprint_payload(payload)[:16]

    @classmethod
    def _resolved_install_lock_hash(
        cls,
        spec: ToolboxEnvironmentSpec,
        *,
        resolved_packages: Sequence[Any],
        source_install_plan_hash: str,
        requirements_relpath: str = "requirements-resolved.txt",
    ) -> str:
        payload = {
            "venv_key": spec.venv_key,
            "environment_name": spec.environment_name,
            "environment_description_hash": spec.environment_description_hash,
            "resolved_packages": cls._unique_names(resolved_packages or []),
            "source_install_plan_hash": str(source_install_plan_hash or "").strip() or None,
            "requirements_relpath": str(requirements_relpath or "").strip() or "requirements-resolved.txt",
            "toolbox_runtime_hash": spec.toolbox_runtime_hash,
            "intrinsics_profile_id": spec.intrinsics_profile_id,
            "dependency_lock_hash": spec.dependency_lock_hash,
            "venv_lock_hash": spec.venv_lock_hash,
        }
        return cls._fingerprint_payload(payload)[:16]

    @classmethod
    def _resolved_packages_from_report(cls, report: Dict[str, Any]) -> List[str]:
        out: List[str] = []
        seen: set[str] = set()
        for item in list(dict(report or {}).get("install") or []):
            row = dict(item or {})
            metadata = dict(row.get("metadata") or {})
            name = str(metadata.get("name") or "").strip()
            version = str(metadata.get("version") or "").strip()
            if not name:
                continue
            pinned = f"{name}=={version}" if version else name
            key = pinned.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(pinned)
        return out

    def read_environment_metadata(self, spec: ToolboxEnvironmentSpec) -> Dict[str, Any]:
        metadata_path = Path(spec.venv_path).expanduser().resolve() / "environment.json"
        if not metadata_path.exists():
            return dict(spec.to_dict())
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            return dict(payload or {}) if isinstance(payload, dict) else dict(spec.to_dict())
        except Exception:
            return dict(spec.to_dict())

    def realize_environment(
        self,
        spec: ToolboxEnvironmentSpec,
        *,
        environment_description: Optional[Dict[str, Any]] = None,
        required_packages: Optional[Sequence[str]] = None,
        missing_packages: Optional[Sequence[str]] = None,
        toolbox_id: Optional[str] = None,
        sandbox_profile_id: Optional[str] = None,
        tool_keys: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        ensured = self.ensure_environment(spec)
        effective_desc_input = dict(environment_description or {})
        effective_desc = {
            "name": str(effective_desc_input.get("name") or ensured.environment_name or "base").strip() or "base",
            "base_env_name": effective_desc_input.get("base_env_name"),
            "effective_extra_packages": self._unique_names(
                effective_desc_input.get("effective_extra_packages")
                or effective_desc_input.get("extra_packages")
                or []
            ),
            "effective_allow_online_install": bool(
                effective_desc_input.get(
                    "effective_allow_online_install",
                    effective_desc_input.get("allow_online_install", False),
                )
            ),
            "lineage": [str(item or "").strip() for item in list(effective_desc_input.get("lineage") or []) if str(item or "").strip()],
        }
        required = self._unique_names(required_packages or ensured.required_imports)
        missing = self._unique_names(missing_packages or [])
        planned = self._unique_names(list(effective_desc["effective_extra_packages"]) + list(required))
        provenance_payload = {
            "toolbox_id": str(toolbox_id or "").strip() or None,
            "sandbox_profile_id": str(sandbox_profile_id or "").strip() or None,
            "venv_key": ensured.venv_key,
            "environment_name": ensured.environment_name,
            "environment_description_hash": ensured.environment_description_hash,
            "required_packages": required,
            "effective_extra_packages": list(effective_desc["effective_extra_packages"]),
            "planned_packages": planned,
            "missing_packages": missing,
            "allow_online_install": bool(effective_desc["effective_allow_online_install"]),
            "tool_keys": self._unique_names(tool_keys or []),
            "dependency_lock_hash": ensured.dependency_lock_hash,
            "venv_lock_hash": ensured.venv_lock_hash,
            "toolbox_runtime_hash": ensured.toolbox_runtime_hash,
            "intrinsics_profile_id": ensured.intrinsics_profile_id,
        }
        provenance_hash = self._fingerprint_payload(provenance_payload)[:16]
        realization = {
            "mode": "metadata_only",
            "status": "planned",
            "provenance_hash": provenance_hash,
            "realized_at": time.time(),
            "required_packages": required,
            "effective_extra_packages": list(effective_desc["effective_extra_packages"]),
            "planned_packages": planned,
            "missing_packages": missing,
            "allow_online_install": bool(effective_desc["effective_allow_online_install"]),
            "environment_lineage": list(effective_desc["lineage"]),
        }
        metadata = self.read_environment_metadata(ensured)
        metadata.update(ensured.to_dict())
        metadata["realization"] = realization
        metadata_path = Path(ensured.venv_path).expanduser().resolve() / "environment.json"
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        return metadata

    def prepare_install_plan(
        self,
        spec: ToolboxEnvironmentSpec,
        *,
        environment_description: Optional[Dict[str, Any]] = None,
        required_packages: Optional[Sequence[str]] = None,
        missing_packages: Optional[Sequence[str]] = None,
        toolbox_id: Optional[str] = None,
        sandbox_profile_id: Optional[str] = None,
        tool_keys: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        metadata = self.realize_environment(
            spec,
            environment_description=environment_description,
            required_packages=required_packages,
            missing_packages=missing_packages,
            toolbox_id=toolbox_id,
            sandbox_profile_id=sandbox_profile_id,
            tool_keys=tool_keys,
        )
        ensured = self.ensure_environment(spec)
        env_root = Path(ensured.venv_path).expanduser().resolve()
        realization = dict(metadata.get("realization") or {})
        planned_packages = self._unique_names(realization.get("planned_packages") or [])
        requirements_relpath = "requirements-planned.txt"
        requirements_path = env_root / requirements_relpath
        requirements_body = "".join(f"{pkg}\n" for pkg in planned_packages)
        requirements_path.write_text(requirements_body, encoding="utf-8")
        install_command = [
            str(ensured.python_executable or self.python_executable_path(env_root)),
            "-m",
            "pip",
            "install",
            "-r",
            str(requirements_path),
        ]
        install_plan = {
            "mode": "plan_only",
            "requirements_path": str(requirements_path),
            "requirements_relpath": requirements_relpath,
            "planned_packages": planned_packages,
            "missing_packages": self._unique_names(realization.get("missing_packages") or []),
            "can_execute_online": bool(realization.get("allow_online_install", False)),
            "install_command": install_command,
            "generated_at": time.time(),
        }
        metadata["install_plan"] = install_plan
        metadata_path = env_root / "environment.json"
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        return metadata

    def lock_install_plan(self, spec: ToolboxEnvironmentSpec) -> Dict[str, Any]:
        ensured = self.ensure_environment(spec)
        env_root = Path(ensured.venv_path).expanduser().resolve()
        metadata = self.read_environment_metadata(ensured)
        install_plan = dict(metadata.get("install_plan") or {})
        if not install_plan:
            raise ValueError("install_plan_missing")
        planned_packages = self._unique_names(install_plan.get("planned_packages") or [])
        requirements_relpath = "requirements-locked.txt"
        requirements_path = env_root / requirements_relpath
        requirements_body = "".join(f"{pkg}\n" for pkg in planned_packages)
        requirements_path.write_text(requirements_body, encoding="utf-8")
        lock_payload = {
            "venv_key": ensured.venv_key,
            "environment_name": ensured.environment_name,
            "environment_description_hash": ensured.environment_description_hash,
            "planned_packages": planned_packages,
            "requirements_relpath": requirements_relpath,
            "toolbox_runtime_hash": ensured.toolbox_runtime_hash,
            "intrinsics_profile_id": ensured.intrinsics_profile_id,
            "dependency_lock_hash": ensured.dependency_lock_hash,
            "venv_lock_hash": ensured.venv_lock_hash,
        }
        install_lock_hash = self._fingerprint_payload(lock_payload)[:16]
        locked_plan = {
            "status": "locked",
            "locked_at": time.time(),
            "install_lock_hash": install_lock_hash,
            "planned_packages": planned_packages,
            "requirements_path": str(requirements_path),
            "requirements_relpath": requirements_relpath,
        }
        metadata["install_lock"] = locked_plan
        (env_root / "environment.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        return metadata

    def resolve_install_lock(
        self,
        spec: ToolboxEnvironmentSpec,
        *,
        allow_resolution: bool = False,
    ) -> Dict[str, Any]:
        ensured = self.ensure_environment(spec)
        env_root = Path(ensured.venv_path).expanduser().resolve()
        metadata = self.read_environment_metadata(ensured)
        install_plan = dict(metadata.get("install_plan") or {})
        if not install_plan:
            raise ValueError("install_plan_missing")
        install_lock = dict(metadata.get("install_lock") or {})
        if not install_lock:
            resolution = {
                "status": "blocked",
                "resolved_at": time.time(),
                "reason": "install_lock_required",
            }
            metadata["install_resolution"] = resolution
            (env_root / "environment.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
            return metadata
        verification_meta = self.verify_install_lock(ensured)
        verification = dict(verification_meta.get("install_lock_verification") or {})
        if str(verification.get("status") or "").strip().lower() != "ok":
            resolution = {
                "status": "blocked",
                "resolved_at": time.time(),
                "reason": str(verification.get("reason") or "install_lock_invalid"),
                "install_lock_hash": str(verification.get("install_lock_hash") or "").strip() or None,
                "expected_install_lock_hash": str(verification.get("expected_install_lock_hash") or "").strip() or None,
            }
            verification_meta["install_resolution"] = resolution
            (env_root / "environment.json").write_text(json.dumps(verification_meta, ensure_ascii=False, indent=2), encoding="utf-8")
            return verification_meta
        metadata = verification_meta
        install_plan = dict(metadata.get("install_plan") or {})
        planned_packages = self._unique_names(install_plan.get("planned_packages") or [])
        if not planned_packages:
            resolution = {
                "status": "noop",
                "resolved_at": time.time(),
                "reason": "no_planned_packages",
            }
            metadata["install_resolution"] = resolution
            (env_root / "environment.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
            return metadata
        if not allow_resolution:
            resolution = {
                "status": "blocked",
                "resolved_at": time.time(),
                "reason": "resolution_not_enabled",
            }
            metadata["install_resolution"] = resolution
            (env_root / "environment.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
            return metadata
        if not bool(install_plan.get("can_execute_online", False)):
            resolution = {
                "status": "blocked",
                "resolved_at": time.time(),
                "reason": "online_resolution_not_allowed",
            }
            metadata["install_resolution"] = resolution
            (env_root / "environment.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
            return metadata

        requirements_path = Path(str(install_plan.get("requirements_path") or "")).expanduser().resolve()
        if not requirements_path.exists():
            raise ValueError("install_plan_requirements_missing")
        report_relpath = "install-resolution-report.json"
        report_path = env_root / report_relpath
        command = [
            str(ensured.python_executable or self.python_executable_path(env_root)),
            "-m",
            "pip",
            "install",
            "--dry-run",
            "--ignore-installed",
            "--report",
            str(report_path),
            "-r",
            str(requirements_path),
        ]
        result = subprocess.run(  # noqa: S603
            command,
            cwd=str(env_root),
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
        resolution = {
            "status": "ok" if int(result.returncode or 0) == 0 else "failed",
            "resolved_at": time.time(),
            "returncode": int(result.returncode or 0),
            "stdout": str(result.stdout or ""),
            "stderr": str(result.stderr or ""),
            "command": command,
            "report_path": str(report_path),
            "report_relpath": report_relpath,
            "source_install_plan_hash": self._install_plan_hash(install_plan),
        }
        metadata["install_resolution"] = resolution
        if resolution["status"] == "ok" and report_path.exists():
            report_text = report_path.read_text(encoding="utf-8")
            report = json.loads(report_text)
            report_hash = _sha256_text(report_text)
            resolved_packages = self._resolved_packages_from_report(dict(report or {}))
            resolved_relpath = "requirements-resolved.txt"
            resolved_path = env_root / resolved_relpath
            resolved_path.write_text("".join(f"{pkg}\n" for pkg in resolved_packages), encoding="utf-8")
            resolved_lock_payload = {
                "venv_key": ensured.venv_key,
                "environment_name": ensured.environment_name,
                "environment_description_hash": ensured.environment_description_hash,
                "resolved_packages": resolved_packages,
                "source_install_plan_hash": resolution["source_install_plan_hash"],
                "requirements_relpath": resolved_relpath,
                "toolbox_runtime_hash": ensured.toolbox_runtime_hash,
                "intrinsics_profile_id": ensured.intrinsics_profile_id,
                "dependency_lock_hash": ensured.dependency_lock_hash,
                "venv_lock_hash": ensured.venv_lock_hash,
            }
            resolved_lock_hash = self._fingerprint_payload(resolved_lock_payload)[:16]
            metadata["resolved_install_lock"] = {
                "status": "locked",
                "locked_at": time.time(),
                "resolved_lock_hash": resolved_lock_hash,
                "resolved_packages": resolved_packages,
                "requirements_path": str(resolved_path),
                "requirements_relpath": resolved_relpath,
                "report_path": str(report_path),
                "report_relpath": report_relpath,
                "report_sha256": report_hash,
                "source_install_plan_hash": resolution["source_install_plan_hash"],
            }
        (env_root / "environment.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        return metadata

    def verify_install_lock(self, spec: ToolboxEnvironmentSpec) -> Dict[str, Any]:
        ensured = self.ensure_environment(spec)
        env_root = Path(ensured.venv_path).expanduser().resolve()
        metadata = self.read_environment_metadata(ensured)
        install_plan = dict(metadata.get("install_plan") or {})
        install_lock = dict(metadata.get("install_lock") or {})
        if not install_plan:
            raise ValueError("install_plan_missing")
        if not install_lock:
            verification = {
                "status": "missing",
                "verified_at": time.time(),
                "reason": "install_lock_missing",
            }
            metadata["install_lock_verification"] = verification
            (env_root / "environment.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
            return metadata

        planned_packages = self._unique_names(install_plan.get("planned_packages") or [])
        expected_requirements_relpath = "requirements-locked.txt"
        expected_payload = {
            "venv_key": ensured.venv_key,
            "environment_name": ensured.environment_name,
            "environment_description_hash": ensured.environment_description_hash,
            "planned_packages": planned_packages,
            "requirements_relpath": expected_requirements_relpath,
            "toolbox_runtime_hash": ensured.toolbox_runtime_hash,
            "intrinsics_profile_id": ensured.intrinsics_profile_id,
            "dependency_lock_hash": ensured.dependency_lock_hash,
            "venv_lock_hash": ensured.venv_lock_hash,
        }
        expected_lock_hash = self._fingerprint_payload(expected_payload)[:16]
        lock_hash = str(install_lock.get("install_lock_hash") or "").strip()
        requirements_path = Path(
            str(install_lock.get("requirements_path") or (env_root / expected_requirements_relpath))
        ).expanduser().resolve()
        requirements_ok = requirements_path.exists()
        status = "ok"
        reason = None
        if not requirements_ok:
            status = "stale"
            reason = "locked_requirements_missing"
        elif lock_hash != expected_lock_hash:
            status = "stale"
            reason = "install_lock_hash_mismatch"
        resolved_install_lock = dict(metadata.get("resolved_install_lock") or {})
        resolved_lock_hash = str(resolved_install_lock.get("resolved_lock_hash") or "").strip()
        expected_resolved_lock_hash = None
        resolved_requirements_path = None
        resolved_report_path = None
        resolved_report_hash = str(resolved_install_lock.get("report_sha256") or "").strip()
        expected_resolved_report_hash = None
        resolved_reason = None
        resolved_status = "missing"
        if resolved_install_lock:
            expected_plan_hash = self._install_plan_hash(install_plan)
            source_plan_hash = str(resolved_install_lock.get("source_install_plan_hash") or "").strip()
            expected_resolved_relpath = (
                str(resolved_install_lock.get("requirements_relpath") or "").strip() or "requirements-resolved.txt"
            )
            expected_resolved_lock_hash = self._resolved_install_lock_hash(
                ensured,
                resolved_packages=resolved_install_lock.get("resolved_packages") or [],
                source_install_plan_hash=expected_plan_hash,
                requirements_relpath=expected_resolved_relpath,
            )
            resolved_requirements_path = Path(
                str(resolved_install_lock.get("requirements_path") or (env_root / expected_resolved_relpath))
            ).expanduser().resolve()
            expected_report_relpath = (
                str(resolved_install_lock.get("report_relpath") or "").strip() or "install-resolution-report.json"
            )
            resolved_report_path = Path(
                str(resolved_install_lock.get("report_path") or (env_root / expected_report_relpath))
            ).expanduser().resolve()
            resolved_status = "ok"
            if source_plan_hash != expected_plan_hash:
                resolved_status = "stale"
                resolved_reason = "resolved_lock_plan_hash_mismatch"
            elif not resolved_requirements_path.exists():
                resolved_status = "stale"
                resolved_reason = "resolved_lock_requirements_missing"
            elif not resolved_report_path.exists():
                resolved_status = "stale"
                resolved_reason = "resolved_lock_report_missing"
            else:
                expected_resolved_report_hash = _sha256_text(resolved_report_path.read_text(encoding="utf-8"))
                if resolved_report_hash and resolved_report_hash != expected_resolved_report_hash:
                    resolved_status = "stale"
                    resolved_reason = "resolved_lock_report_hash_mismatch"
            if resolved_status == "ok" and resolved_lock_hash != expected_resolved_lock_hash:
                resolved_status = "stale"
                resolved_reason = "resolved_lock_hash_mismatch"
            if resolved_status != "ok":
                status = "stale"
                reason = resolved_reason
        verification = {
            "status": status,
            "verified_at": time.time(),
            "install_lock_hash": lock_hash or None,
            "expected_install_lock_hash": expected_lock_hash,
            "requirements_path": str(requirements_path),
            "reason": reason,
            "resolved_lock_status": resolved_status,
            "resolved_lock_hash": resolved_lock_hash or None,
            "expected_resolved_lock_hash": expected_resolved_lock_hash,
            "resolved_requirements_path": str(resolved_requirements_path) if resolved_requirements_path else None,
            "resolved_report_path": str(resolved_report_path) if resolved_report_path else None,
            "resolved_report_sha256": resolved_report_hash or None,
            "expected_resolved_report_sha256": expected_resolved_report_hash,
            "resolved_reason": resolved_reason,
        }
        metadata["install_lock_verification"] = verification
        (env_root / "environment.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        return metadata

    def execute_install_plan(
        self,
        spec: ToolboxEnvironmentSpec,
        *,
        allow_execution: bool = False,
    ) -> Dict[str, Any]:
        ensured = self.ensure_environment(spec)
        env_root = Path(ensured.venv_path).expanduser().resolve()
        metadata = self.read_environment_metadata(ensured)
        install_plan = dict(metadata.get("install_plan") or {})
        if not install_plan:
            raise ValueError("install_plan_missing")
        planned_packages = self._unique_names(install_plan.get("planned_packages") or [])
        if not planned_packages:
            execution = {
                "status": "noop",
                "executed": False,
                "executed_at": time.time(),
                "reason": "no_planned_packages",
            }
            metadata["install_execution"] = execution
            (env_root / "environment.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
            return metadata
        if not allow_execution:
            execution = {
                "status": "blocked",
                "executed": False,
                "executed_at": time.time(),
                "reason": "execution_not_enabled",
            }
            metadata["install_execution"] = execution
            (env_root / "environment.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
            return metadata
        if not bool(install_plan.get("can_execute_online", False)):
            execution = {
                "status": "blocked",
                "executed": False,
                "executed_at": time.time(),
                "reason": "online_install_not_allowed",
            }
            metadata["install_execution"] = execution
            (env_root / "environment.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
            return metadata

        verification_meta = self.verify_install_lock(ensured)
        verification = dict(verification_meta.get("install_lock_verification") or {})
        if str(verification.get("status") or "") != "ok":
            verification_reason = str(verification.get("reason") or "").strip()
            execution = {
                "status": "blocked",
                "executed": False,
                "executed_at": time.time(),
                "reason": (
                    "install_lock_required"
                    if verification_reason in {"", "install_lock_missing"}
                    else verification_reason
                ),
                "install_lock_hash": str(verification.get("install_lock_hash") or "").strip() or None,
                "expected_install_lock_hash": str(verification.get("expected_install_lock_hash") or "").strip() or None,
            }
            metadata = self.read_environment_metadata(ensured)
            metadata["install_execution"] = execution
            (env_root / "environment.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
            return metadata

        metadata = verification_meta
        install_lock = dict(metadata.get("install_lock") or {})
        resolved_install_lock = dict(metadata.get("resolved_install_lock") or {})
        requirements_path = str(
            install_lock.get("requirements_path")
            or install_plan.get("requirements_path")
            or ""
        ).strip()
        command = [str(item or "").strip() for item in list(install_plan.get("install_command") or []) if str(item or "").strip()]
        if not command:
            raise ValueError("install_command_missing")
        resolved_lock_hash = None
        if resolved_install_lock:
            expected_plan_hash = self._install_plan_hash(install_plan)
            source_plan_hash = str(resolved_install_lock.get("source_install_plan_hash") or "").strip()
            resolved_requirements_path = Path(
                str(resolved_install_lock.get("requirements_path") or "")
            ).expanduser().resolve()
            if source_plan_hash != expected_plan_hash:
                execution = {
                    "status": "blocked",
                    "executed": False,
                    "executed_at": time.time(),
                    "reason": "resolved_lock_plan_hash_mismatch",
                    "resolved_lock_hash": str(resolved_install_lock.get("resolved_lock_hash") or "").strip() or None,
                    "source_install_plan_hash": source_plan_hash or None,
                    "expected_install_plan_hash": expected_plan_hash,
                }
                metadata["install_execution"] = execution
                (env_root / "environment.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
                return metadata
            if resolved_requirements_path.exists():
                requirements_path = str(resolved_requirements_path)
                resolved_lock_hash = str(resolved_install_lock.get("resolved_lock_hash") or "").strip() or None
        if requirements_path:
            command = command[:-1] + [requirements_path]
        result = subprocess.run(  # noqa: S603
            command,
            cwd=str(env_root),
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
        execution = {
            "status": "ok" if int(result.returncode or 0) == 0 else "failed",
            "executed": True,
            "executed_at": time.time(),
            "returncode": int(result.returncode or 0),
            "stdout": str(result.stdout or ""),
            "stderr": str(result.stderr or ""),
            "command": command,
            "install_lock_hash": str(install_lock.get("install_lock_hash") or "").strip() or None,
            "resolved_lock_hash": resolved_lock_hash,
        }
        metadata["install_execution"] = execution
        if execution["status"] == "ok":
            freeze_cmd = [
                str(ensured.python_executable or self.python_executable_path(env_root)),
                "-m",
                "pip",
                "freeze",
            ]
            try:
                freeze_result = subprocess.run(  # noqa: S603
                    freeze_cmd,
                    cwd=str(env_root),
                    capture_output=True,
                    text=True,
                    timeout=120,
                    check=False,
                )
                freeze_output = str(freeze_result.stdout or "")
                lines = [
                    line.strip()
                    for line in freeze_output.splitlines()
                    if str(line or "").strip()
                ]
                receipt_payload = {
                    "status": "ok" if int(freeze_result.returncode or 0) == 0 else "failed",
                    "captured_at": time.time(),
                    "returncode": int(freeze_result.returncode or 0),
                    "command": freeze_cmd,
                    "packages": lines,
                    "packages_hash": self._fingerprint_payload({"packages": lines})[:16],
                    "stderr": str(freeze_result.stderr or ""),
                }
            except Exception as exc:
                receipt_payload = {
                    "status": "failed",
                    "captured_at": time.time(),
                    "command": freeze_cmd,
                    "packages": [],
                    "packages_hash": None,
                    "stderr": str(exc),
                }
            metadata["install_receipt"] = receipt_payload
            (env_root / "environment.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
            metadata = self.verify_install_receipt(ensured)
        (env_root / "environment.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        return metadata

    def verify_install_receipt(self, spec: ToolboxEnvironmentSpec) -> Dict[str, Any]:
        ensured = self.ensure_environment(spec)
        env_root = Path(ensured.venv_path).expanduser().resolve()
        metadata = self.verify_install_lock(ensured)
        install_lock = dict(metadata.get("install_lock") or {})
        resolved_install_lock = dict(metadata.get("resolved_install_lock") or {})
        install_receipt = dict(metadata.get("install_receipt") or {})
        lock_verification = dict(metadata.get("install_lock_verification") or {})
        if not install_lock and not resolved_install_lock:
            verification = {
                "status": "missing",
                "verified_at": time.time(),
                "reason": "install_lock_missing",
            }
            metadata["install_receipt_verification"] = verification
            (env_root / "environment.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
            return metadata
        if not install_receipt:
            verification = {
                "status": "missing",
                "verified_at": time.time(),
                "reason": "install_receipt_missing",
            }
            metadata["install_receipt_verification"] = verification
            (env_root / "environment.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
            return metadata
        if str(lock_verification.get("status") or "").strip() not in {"ok", "missing"}:
            verification = {
                "status": "stale",
                "verified_at": time.time(),
                "reason": str(lock_verification.get("reason") or "install_lock_invalid"),
                "lock_verification_status": str(lock_verification.get("status") or "").strip() or None,
                "lock_source": "resolved_install_lock" if resolved_install_lock else "install_lock",
            }
            metadata["install_receipt_verification"] = verification
            (env_root / "environment.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
            return metadata

        locked_source = list(resolved_install_lock.get("resolved_packages") or []) or list(install_lock.get("planned_packages") or [])
        locked_names = {
            self._normalize_package_name(item)
            for item in locked_source
            if self._normalize_package_name(item)
        }
        observed_names = {
            self._normalize_package_name(item)
            for item in list(install_receipt.get("packages") or [])
            if self._normalize_package_name(item)
        }
        missing = sorted(name for name in locked_names if name not in observed_names)
        status = "ok" if not missing else "mismatch"
        verification = {
            "status": status,
            "verified_at": time.time(),
            "locked_package_names": sorted(locked_names),
            "observed_package_names": sorted(observed_names),
            "missing_package_names": missing,
            "lock_source": "resolved_install_lock" if resolved_install_lock else "install_lock",
        }
        metadata["install_receipt_verification"] = verification
        (env_root / "environment.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        return metadata

    def ensure_for_bundle(
        self,
        staged: "StagedToolboxBundle",
        *,
        environment_description: Optional[Dict[str, Any]] = None,
    ) -> ToolboxEnvironmentSpec:
        return self.ensure_environment(self.environment_spec_for_bundle(staged, environment_description=environment_description))


@dataclass
class StagedToolboxBundle:
    bundle_root: Path
    manifest_path: Path
    manifest: Dict[str, Any]

    def registration_bundle(self) -> Dict[str, Any]:
        return {
            "bundle_id": str(self.manifest.get("bundle_id") or ""),
            "toolbox_id": str(self.manifest.get("toolbox_id") or self.manifest.get("bundle_id") or ""),
            "sandbox_profile_id": str(dict(self.manifest.get("sandbox_profile") or {}).get("profile_id") or "default"),
            "bundle_revision": str(self.manifest.get("bundle_revision") or ""),
            "manifest_hash": str(self.manifest.get("manifest_hash") or ""),
            "bundle_root": str(self.bundle_root),
            "manifest_path": str(self.manifest_path),
        }

    def registration_environment(self, environment_spec: Optional[ToolboxEnvironmentSpec] = None) -> Dict[str, Any]:
        spec = environment_spec
        if spec is None:
            spec = ToolboxEnvironmentManager(self.bundle_root.parents[2]).environment_spec_for_bundle(self)
        return {
            "venv_key": spec.venv_key,
            "venv_path": spec.venv_path,
            "python_executable": spec.python_executable,
            "environment_name": spec.environment_name,
            "environment_description_hash": spec.environment_description_hash,
            "venv_lock_hash": spec.venv_lock_hash,
            "venv_mutable": False,
            "toolbox_runtime_hash": spec.toolbox_runtime_hash,
            "intrinsics_profile_id": spec.intrinsics_profile_id,
            "required_imports": list(spec.required_imports or []),
            "dependency_lock_hash": spec.dependency_lock_hash,
        }

    def registration_tool_access(self) -> Dict[str, Any]:
        tool_names = [str(item.get("name") or "").strip() for item in list(self.manifest.get("tools") or [])]
        tool_names = [name for name in tool_names if name]
        auto_tool_names = [str(item.get("name") or "").strip() for item in list(self.manifest.get("auto_tools") or [])]
        for name in auto_tool_names:
            if name and name not in tool_names:
                tool_names.append(name)
        hidden_tool_names = {
            str(item or "").strip()
            for item in list(self.manifest.get("hidden_tool_names") or [])
            if str(item or "").strip()
        }
        active_intrinsic_names = [
            str(item or "").strip()
            for item in list(self.manifest.get("active_intrinsic_tool_names") or [])
            if str(item or "").strip()
        ]
        hidden_intrinsic_names = {
            str(item or "").strip()
            for item in list(self.manifest.get("hidden_intrinsic_tool_names") or [])
            if str(item or "").strip()
        }
        allowed_tool_names = list(tool_names)
        for name in active_intrinsic_names:
            if name not in allowed_tool_names:
                allowed_tool_names.append(name)
        hidden_allowed_tool_names = [name for name in allowed_tool_names if name in hidden_tool_names or name in hidden_intrinsic_names]
        advertised_tool_names = [name for name in allowed_tool_names if name not in set(hidden_allowed_tool_names)]
        sandbox_profile_id = str(dict(self.manifest.get("sandbox_profile") or {}).get("profile_id") or "default")
        return {
            "allowed_tool_names": allowed_tool_names,
            "advertised_tool_names": advertised_tool_names,
            "hidden_allowed_tool_names": hidden_allowed_tool_names,
            "tool_routes": {
                name: {
                    "toolbox_id": str(self.manifest.get("toolbox_id") or self.manifest.get("bundle_id") or ""),
                    "sandbox_profile_id": sandbox_profile_id,
                }
                for name in allowed_tool_names
            },
        }

    def worker_command(self, *, python_executable: Optional[str] = None) -> List[str]:
        return [
            str(python_executable or sys.executable),
            "-m",
            "hosting.toolbox_executor_ipc",
        ]

    def worker_startup_spec(
        self,
        *,
        worker_id: str,
        sandbox_id: Optional[str] = None,
        scratch_root: Optional[Path] = None,
        engines_state_file: Optional[Path] = None,
        control_state_file: Optional[Path] = None,
        venv_path: Optional[str] = None,
        ipc_family: Optional[str] = None,
        ipc_address: str = "",
        policy: Optional[Dict[str, Any]] = None,
    ) -> ToolboxWorkerStartupSpec:
        scratch = Path(scratch_root or (self.bundle_root / "scratch")).expanduser().resolve()
        default_ipc_family = "AF_PIPE" if os.name == "nt" else "AF_UNIX"
        return ToolboxWorkerStartupSpec(
            worker_id=str(worker_id or "").strip(),
            sandbox_id=str(sandbox_id or worker_id or "").strip(),
            toolbox_revision=str(self.manifest.get("bundle_revision") or "").strip(),
            manifest_path=str(self.manifest_path),
            scratch_root=str(scratch),
            engines_state_file=str(Path(engines_state_file).expanduser().resolve()) if engines_state_file else None,
            control_state_file=str(Path(control_state_file).expanduser().resolve()) if control_state_file else None,
            venv_path=str(venv_path or "").strip() or None,
            ipc_family=str(ipc_family or default_ipc_family).strip() or default_ipc_family,
            ipc_address=str(ipc_address or "").strip(),
            policy=dict(policy or {}),
        )

    def worker_env(self, *, extra_env: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        src_root = str(Path(__file__).resolve().parents[1])
        env = {str(k): str(v) for k, v in dict(extra_env or {}).items()}
        env["MP13_TOOLBOX_MANIFEST_PATH"] = str(self.manifest_path)
        current_py = str(env.get("PYTHONPATH") or "")
        paths = [p for p in current_py.split(os.pathsep) if p] if current_py else []
        if src_root not in paths:
            env["PYTHONPATH"] = src_root if not current_py else f"{src_root}{os.pathsep}{current_py}"
        return env

    def worker_env_with_startup_spec(
        self,
        *,
        worker_id: str,
        sandbox_id: Optional[str] = None,
        scratch_root: Optional[Path] = None,
        engines_state_file: Optional[Path] = None,
        control_state_file: Optional[Path] = None,
        venv_path: Optional[str] = None,
        ipc_family: Optional[str] = None,
        ipc_address: str = "",
        policy: Optional[Dict[str, Any]] = None,
        extra_env: Optional[Dict[str, str]] = None,
    ) -> Dict[str, str]:
        env = self.worker_env(extra_env=extra_env)
        spec = self.worker_startup_spec(
            worker_id=worker_id,
            sandbox_id=sandbox_id,
            scratch_root=scratch_root,
            engines_state_file=engines_state_file,
            control_state_file=control_state_file,
            venv_path=venv_path,
            ipc_family=ipc_family,
            ipc_address=ipc_address,
            policy=policy,
        )
        fd, tmp_name = tempfile.mkstemp(
            prefix=f"mp13-toolbox-startup-{spec.worker_id or 'worker'}-",
            suffix=".json",
            dir=str(self.bundle_root),
        )
        os.close(fd)
        spec.write_json(Path(tmp_name))
        env["MP13_TOOLBOX_WORKER_SPEC_PATH"] = str(Path(tmp_name).resolve())
        return env


class ToolboxBundleStager:
    def __init__(self, hosting_root: Path):
        self.hosting_root = Path(hosting_root).expanduser().resolve()

    def stage_bundle(self, spec: ToolboxBundleSpec) -> StagedToolboxBundle:
        manifest = spec.manifest_payload()
        bundle_root = (
            self.hosting_root
            / "toolbox_bundles"
            / str(manifest["bundle_id"])
            / str(manifest["bundle_revision"])
        ).resolve()
        files_root = (bundle_root / "files").resolve()
        files_root.mkdir(parents=True, exist_ok=True)
        for file_spec in spec.files:
            rel = file_spec.normalized_path()
            target = (files_root / rel).resolve()
            if files_root not in target.parents and target != files_root:
                raise ValueError("bundle_file_path_invalid")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(str(file_spec.content or ""), encoding="utf-8")
        manifest_path = bundle_root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return StagedToolboxBundle(bundle_root=bundle_root, manifest_path=manifest_path, manifest=manifest)


class ToolboxSandboxOrchestrator:
    def __init__(
        self,
        *,
        service: Any,
        stager: ToolboxBundleStager,
        python_executable: Optional[str] = None,
    ) -> None:
        self.service = service
        self.stager = stager
        self.python_executable = str(python_executable or sys.executable)
        self.environment_manager = ToolboxEnvironmentManager(self.stager.hosting_root)

    @staticmethod
    def _bundle_id(toolbox_id: str, profile: SandboxProfileSpec) -> str:
        return f"{str(toolbox_id or '').strip()}-{profile.normalized_profile_id()}"

    @staticmethod
    def _engine_id(toolbox_id: str, profile: SandboxProfileSpec, revision: str) -> str:
        return f"{str(toolbox_id or '').strip()}-{profile.normalized_profile_id()}-{str(revision or '')[:8]}"

    @staticmethod
    def _capabilities_for_profile(profile: SandboxProfileSpec) -> Dict[str, Any]:
        brokered = dict(dict(profile.sandbox_policy or {}).get("sandbox") or {}).get("brokered_io")
        return {
            "brokered_filesystem": bool(dict(brokered or {}).get("filesystem", False)),
            "brokered_http": bool(dict(brokered or {}).get("http", False)),
            "dynamic_reload": False,
        }

    def build_assignments(
        self,
        *,
        toolbox_id: str,
        requests: Sequence[ToolboxAutoAssignmentRequest],
        manual_requests: Optional[Sequence[ToolboxManualAssignmentRequest]] = None,
        intrinsic_tool_names: Optional[Sequence[str]] = None,
        intrinsic_profile: Optional[SandboxProfileSpec] = None,
        with_intrinsic_guides: bool = False,
    ) -> List[ToolboxSandboxAssignment]:
        tid = str(toolbox_id or "").strip()
        if not tid:
            raise ValueError("toolbox_id_required")
        grouped: Dict[str, Dict[str, Any]] = {}
        for request in list(requests or []):
            profile = request.sandbox_profile or SandboxProfileSpec()
            profile_key = profile.normalized_profile_id()
            row = grouped.setdefault(profile_key, {"profile": profile, "files": [], "auto_tools": [], "tools": []})
            row["files"].extend(list(request.files or []))
            row["auto_tools"].append(request.to_auto_tool())
        for request in list(manual_requests or []):
            profile = request.sandbox_profile or SandboxProfileSpec()
            profile_key = profile.normalized_profile_id()
            row = grouped.setdefault(profile_key, {"profile": profile, "files": [], "auto_tools": [], "tools": []})
            row["files"].extend(list(request.files or []))
            row["tools"].append(request.to_bundle_tool())
        out: List[ToolboxSandboxAssignment] = []
        for row in grouped.values():
            profile = row["profile"]
            file_map: Dict[str, ToolboxBundleFile] = {}
            for file_spec in list(row["files"] or []):
                file_map[file_spec.normalized_path()] = file_spec
            hidden_tool_names: List[str] = []
            for tool_spec in list(row["tools"] or []):
                if bool(getattr(tool_spec, "hidden", False)):
                    name = tool_spec.tool_name()
                    if name not in hidden_tool_names:
                        hidden_tool_names.append(name)
            for auto_spec in list(row["auto_tools"] or []):
                if bool(getattr(auto_spec, "hidden", False)):
                    name = auto_spec.tool_name()
                    if name not in hidden_tool_names:
                        hidden_tool_names.append(name)
            spec = ToolboxBundleSpec(
                bundle_id=self._bundle_id(tid, profile),
                toolbox_id=tid,
                sandbox_profile=profile,
                files=list(file_map.values()),
                tools=list(row["tools"] or []),
                auto_tools=list(row["auto_tools"] or []),
                hidden_tool_names=hidden_tool_names,
            )
            out.append(
                ToolboxSandboxAssignment(
                    toolbox_id=tid,
                    sandbox_profile=profile,
                    bundle_spec=spec,
                )
            )
        intrinsic_names = [str(item or "").strip() for item in list(intrinsic_tool_names or []) if str(item or "").strip()]
        if intrinsic_names:
            profile = intrinsic_profile or SandboxProfileSpec(profile_id="default")
            profile_id = profile.normalized_profile_id()
            existing = next((item for item in out if item.sandbox_profile.normalized_profile_id() == profile_id), None)
            if existing is None:
                existing = ToolboxSandboxAssignment(
                    toolbox_id=tid,
                    sandbox_profile=profile,
                    bundle_spec=ToolboxBundleSpec(
                        bundle_id=self._bundle_id(tid, profile),
                        toolbox_id=tid,
                        sandbox_profile=profile,
                    ),
                )
                out.append(existing)
            existing.bundle_spec.with_intrinsics = True
            existing.bundle_spec.with_intrinsic_guides = bool(with_intrinsic_guides)
            existing.bundle_spec.intrinsic_tool_names = intrinsic_names
            existing.bundle_spec.active_intrinsic_tool_names = intrinsic_names
        return sorted(out, key=lambda item: item.sandbox_profile.normalized_profile_id())

    def stage_assignments(
        self,
        *,
        toolbox_id: str,
        requests: Sequence[ToolboxAutoAssignmentRequest],
        manual_requests: Optional[Sequence[ToolboxManualAssignmentRequest]] = None,
        intrinsic_tool_names: Optional[Sequence[str]] = None,
        intrinsic_profile: Optional[SandboxProfileSpec] = None,
        with_intrinsic_guides: bool = False,
    ) -> List[ToolboxSandboxAssignment]:
        assignments = self.build_assignments(
            toolbox_id=toolbox_id,
            requests=requests,
            manual_requests=manual_requests,
            intrinsic_tool_names=intrinsic_tool_names,
            intrinsic_profile=intrinsic_profile,
            with_intrinsic_guides=with_intrinsic_guides,
        )
        for item in assignments:
            item.staged_bundle = self.stager.stage_bundle(item.bundle_spec)
        return assignments

    def spawn_assignments(
        self,
        *,
        toolbox_id: str,
        requests: Sequence[ToolboxAutoAssignmentRequest],
        manual_requests: Optional[Sequence[ToolboxManualAssignmentRequest]] = None,
        intrinsic_tool_names: Optional[Sequence[str]] = None,
        intrinsic_profile: Optional[SandboxProfileSpec] = None,
        with_intrinsic_guides: bool = False,
        worker_profile_class: str = "generic",
    ) -> List[ToolboxSandboxAssignment]:
        assignments = self.stage_assignments(
            toolbox_id=toolbox_id,
            requests=requests,
            manual_requests=manual_requests,
            intrinsic_tool_names=intrinsic_tool_names,
            intrinsic_profile=intrinsic_profile,
            with_intrinsic_guides=with_intrinsic_guides,
        )
        for item in assignments:
            if item.staged_bundle is None:
                raise RuntimeError("staged_bundle_required")
            staged = item.staged_bundle
            revision = str(staged.manifest.get("bundle_revision") or "")
            engine_id = self._engine_id(toolbox_id, item.sandbox_profile, revision)
            environment_name = str(item.sandbox_profile.environment_name or "base").strip() or "base"
            environment_description = None
            if hasattr(self.service, "toolbox_environment_description_effective_get"):
                try:
                    environment_description = self.service.toolbox_environment_description_effective_get(environment_name)
                except Exception:
                    environment_description = None
            elif hasattr(self.service, "toolbox_environment_description_get"):
                try:
                    environment_description = self.service.toolbox_environment_description_get(environment_name)
                except Exception:
                    environment_description = None
            environment_spec = self.environment_manager.ensure_for_bundle(
                staged,
                environment_description=environment_description,
            )
            environment_spec.python_executable = self.environment_manager.runtime_python_executable(
                environment_spec,
                fallback_python_executable=self.python_executable,
            )
            item.registration = self.service.spawn(
                engine_id=engine_id,
                command=staged.worker_command(
                    python_executable=environment_spec.python_executable or self.python_executable
                ),
                env=staged.worker_env_with_startup_spec(
                    worker_id=engine_id,
                    sandbox_id=f"{str(toolbox_id or '').strip()}-{item.sandbox_profile.normalized_profile_id()}",
                    scratch_root=self.stager.hosting_root / "toolbox_scratch" / engine_id,
                    engines_state_file=self.service.engines_state_file,
                    control_state_file=self.service.control_state_file,
                    venv_path=environment_spec.venv_path,
                    policy=dict(item.sandbox_profile.sandbox_policy or {}),
                ),
                worker_profile_class=worker_profile_class,
                sandbox_policy=dict(item.sandbox_profile.sandbox_policy or {}),
                executor_kind="toolbox_executor_v1",
                bundle=staged.registration_bundle(),
                environment=staged.registration_environment(environment_spec),
                tool_access=staged.registration_tool_access(),
                capabilities=self._capabilities_for_profile(item.sandbox_profile),
            )
        return assignments


@dataclass
class ToolboxHarnessConfig:
    mode: str = "native"
    sandbox_toolbox_id: Optional[str] = None
    sandbox_engine_ids: List[str] = field(default_factory=list)
    sandbox_selection: str = "round_robin"


class ToolboxExecutionHarness:
    def __init__(
        self,
        *,
        config: Optional[ToolboxHarnessConfig] = None,
        native_toolbox: Optional[Toolbox] = None,
        control_channel: Optional[Any] = None,
    ) -> None:
        self.config = config or ToolboxHarnessConfig()
        self.native_toolbox = native_toolbox
        self.control_channel = control_channel
        self._rr_index = 0
        self._rr_lock = asyncio.Lock()

    async def describe(self) -> Dict[str, Any]:
        mode = str(self.config.mode or "native").strip().lower()
        if mode == "native":
            if self.native_toolbox is None:
                raise RuntimeError("native_toolbox_not_configured")
            names = sorted(list(self.native_toolbox._registered_tool_names()))
            return {
                "mode": "native",
                "executor_kind": "native_toolbox",
                "all_registered_tool_names": names,
                "parallel_execution": {
                    "async_within_executor": True,
                    "sandbox_pool": False,
                },
            }
        engine_id = await self._select_engine_id()
        toolbox_id = str(self.config.sandbox_toolbox_id or "").strip()
        if toolbox_id:
            result = await asyncio.to_thread(self.control_channel.toolbox_describe, toolbox_id=toolbox_id)
        else:
            result = await asyncio.to_thread(self.control_channel.toolbox_describe, engine_id=engine_id)
        out = dict(result or {})
        out.setdefault("mode", "sandbox")
        out.setdefault(
            "parallel_execution",
            {
                "async_within_executor": True,
                "sandbox_pool": len(self.config.sandbox_engine_ids) > 1,
            },
        )
        return out

    async def execute_calls(
        self,
        tool_calls: Sequence[ToolCall | Dict[str, Any]],
        *,
        parallel: bool = True,
        timeout_seconds: float = 30.0,
        native_execute_kwargs: Optional[Dict[str, Any]] = None,
        callback_processor: Optional[Callable[..., Any]] = None,
        callback_context: Any = None,
    ) -> List[ToolCall]:
        calls = [item if isinstance(item, ToolCall) else ToolCall.from_dict(dict(item or {})) for item in list(tool_calls or [])]
        if not calls:
            return []
        if not parallel:
            out: List[ToolCall] = []
            for call in calls:
                out.append(
                    await self._execute_one(
                        call,
                        timeout_seconds=timeout_seconds,
                        native_execute_kwargs=dict(native_execute_kwargs or {}),
                        callback_processor=callback_processor,
                        callback_context=callback_context,
                    )
                )
            return out
        tasks = [
            self._execute_one(
                call,
                timeout_seconds=timeout_seconds,
                native_execute_kwargs=dict(native_execute_kwargs or {}),
                callback_processor=callback_processor,
                callback_context=callback_context,
            )
            for call in calls
        ]
        return list(await asyncio.gather(*tasks))

    async def execute_request_tools(
        self,
        parser_profile: ParserProfile,
        final_response_items: List[InferenceResponse],
        action_handler: Callable[..., Any],
        serial_execution: bool = False,
        *,
        tools_view: Optional[ToolsView] = None,
        context: Optional[Any] = None,
        tool_retries_max: Optional[int] = None,
        tool_retries_left: Optional[int] = None,
        timeout_seconds: float = 30.0,
        callback_processor: Optional[Callable[..., Any]] = None,
        callback_context: Any = None,
        **kwargs: Any,
    ) -> None:
        mode = str(self.config.mode or "native").strip().lower()
        if mode == "native" and self.native_toolbox is not None:
            await self.native_toolbox.execute_request_tools(
                parser_profile=parser_profile,
                final_response_items=final_response_items,
                action_handler=action_handler,
                serial_execution=serial_execution,
                tools_view=tools_view,
                context=context,
                tool_retries_max=tool_retries_max,
                tool_retries_left=tool_retries_left,
                **kwargs,
            )
            return

        all_blocks_to_parse: List[ToolCallBlock] = []
        for response_item in list(final_response_items or []):
            if response_item.tool_blocks and len(response_item.tool_blocks) > 0:
                for block in response_item.tool_blocks:
                    if block.prompt_index is None:
                        block.prompt_index = response_item.prompt_index
                all_blocks_to_parse.extend(response_item.tool_blocks)

        if not all_blocks_to_parse:
            return

        parser = UnifiedToolIO(profile=parser_profile)
        parser.parse_collected_blocks(all_blocks_to_parse)

        parsed_kwargs: Dict[str, Any] = {
            **kwargs,
            "context": context,
            "final_response_items": final_response_items,
            "current_response_item": None,
            "parser": parser,
            "tool_call": None,
            "tool_call_block": None,
            "tools_view": tools_view,
            "tool_retries_max": tool_retries_max,
            "tool_retries_left": tool_retries_left,
            "serial_execution": serial_execution,
        }
        await action_handler(execute_stage="calls_parsed", **parsed_kwargs)

        async def _execute_and_handle(
            tool_call: ToolCall,
            *,
            response_item: InferenceResponse,
            block: ToolCallBlock,
        ) -> None:
            action_kwargs = {
                **kwargs,
                "context": context,
                "final_response_items": final_response_items,
                "current_response_item": response_item,
                "parser": parser,
                "tool_call_block": block,
                "tools_view": tools_view,
                "tool_retries_max": tool_retries_max,
                "tool_retries_left": tool_retries_left,
                "serial_execution": serial_execution,
            }
            try:
                await action_handler(execute_stage="call_starting", tool_call=tool_call, **action_kwargs)
                executed = await self._execute_one(
                    tool_call,
                    timeout_seconds=float(timeout_seconds or 30.0),
                    native_execute_kwargs=dict(
                        kwargs,
                        context=context,
                        tools_view=tools_view,
                        tool_retries_max=tool_retries_max,
                        tool_retries_left=tool_retries_left,
                    ),
                    callback_processor=callback_processor,
                    callback_context=callback_context,
                )
                tool_call.result = executed.result
                tool_call.error = executed.error
                tool_call.action = list(executed.action or [])
                tool_call.id = executed.id or tool_call.id
                tool_call.parse_errors = list(executed.parse_errors or tool_call.parse_errors or [])
                tool_call.raw = executed.raw or tool_call.raw
                tool_call.model_format = executed.model_format or tool_call.model_format
            except Exception as exc:
                if not tool_call.error:
                    tool_call.error = f"Execution failed: {type(exc).__name__} - {exc}"
            finally:
                await action_handler(execute_stage="call_finished", tool_call=tool_call, **action_kwargs)

        if serial_execution:
            for response_item in list(final_response_items or []):
                for block in list(response_item.tool_blocks or []):
                    if not block.calls and not block.is_incomplete:
                        block.error_block = "Tool calls list is empty."
                        if ToolCall.KeepRaw not in (block.action_block or []):
                            block.action_block = list(block.action_block or [])
                            block.action_block.append(ToolCall.KeepRaw)
                        continue
                    if ToolCall.Ignore in block.action_block:
                        continue
                    for tool_call in list(block.calls or []):
                        if ToolCall.Ignore in tool_call.action:
                            continue
                        await _execute_and_handle(tool_call, response_item=response_item, block=block)
        else:
            tasks: List[asyncio.Task[Any]] = []
            for response_item in list(final_response_items or []):
                for block in list(response_item.tool_blocks or []):
                    if not block.calls and not block.is_incomplete:
                        block.error_block = "Tool calls list is empty."
                        if ToolCall.KeepRaw not in (block.action_block or []):
                            block.action_block = list(block.action_block or [])
                            block.action_block.append(ToolCall.KeepRaw)
                        continue
                    if ToolCall.Ignore in block.action_block:
                        continue
                    for tool_call in list(block.calls or []):
                        if ToolCall.Ignore in tool_call.action:
                            continue
                        tasks.append(asyncio.create_task(_execute_and_handle(tool_call, response_item=response_item, block=block)))
            if tasks:
                await asyncio.gather(*tasks)

        await action_handler(execute_stage="all_finished", **parsed_kwargs)

    async def _execute_one(
        self,
        call: ToolCall,
        *,
        timeout_seconds: float,
        native_execute_kwargs: Dict[str, Any],
        callback_processor: Optional[Callable[..., Any]] = None,
        callback_context: Any = None,
    ) -> ToolCall:
        mode = str(self.config.mode or "native").strip().lower()
        if mode == "native":
            if self.native_toolbox is None:
                raise RuntimeError("native_toolbox_not_configured")
            result = await self.native_toolbox.execute(call, **dict(native_execute_kwargs or {}))
            if result is not None:
                call.result = result
            return call
        engine_id = await self._select_engine_id()
        toolbox_id = str(self.config.sandbox_toolbox_id or "").strip()
        tools_view_payload = serialize_tools_view(native_execute_kwargs.get("tools_view"))
        gate_payload: Dict[str, Any] = {}
        if hasattr(self.control_channel, "toolbox_gate"):
            if toolbox_id:
                gate_payload = dict(
                    await asyncio.to_thread(
                        self.control_channel.toolbox_gate,
                        toolbox_id=toolbox_id,
                        tool_name=str(call.name or "").strip(),
                        tools_view=tools_view_payload,
                    )
                )
            else:
                gate_payload = dict(
                    await asyncio.to_thread(
                        self.control_channel.toolbox_gate,
                        engine_id=engine_id,
                        tool_name=str(call.name or "").strip(),
                        tools_view=tools_view_payload,
                    )
                )
        outcome = str(gate_payload.get("outcome") or "").strip().lower()
        if outcome and outcome != "allowed":
            reason = str(gate_payload.get("reason") or outcome).strip() or outcome
            call.error = f"Execution gated: {outcome} - {reason}:{str(call.name or '').strip()}"
            return call
        try:
            callback_binding = None
            if callable(callback_processor):
                if not hasattr(self, "_callback_relay"):
                    self._callback_relay = _HostedToolCallbackRelay()
                signature = None
                try:
                    described = await self.describe()
                    tool_meta = dict(described.get("tool_metadata") or {}).get(str(call.name or "").strip()) or {}
                    signature = dict(tool_meta.get("callback_signature") or {}) or None
                except Exception:
                    signature = None
                callback_binding = self._callback_relay.bind_session(
                    processor=callback_processor,
                    toolbox_id=toolbox_id,
                    tool_name=str(call.name or "").strip(),
                    tool_call_id=str(call.id or "").strip(),
                    tool_arguments=dict(call.arguments or {}),
                    callback_signature=signature,
                    user_context=callback_context,
                )
            if toolbox_id:
                rpc_out = await asyncio.to_thread(
                    self.control_channel.toolbox_execute,
                    toolbox_id=toolbox_id,
                    tool_call=call.to_dict(),
                    timeout_seconds=float(timeout_seconds or 30.0),
                    tools_view=tools_view_payload,
                    callback_binding=dict(callback_binding or {}) or None,
                )
            else:
                rpc_out = await asyncio.to_thread(
                    self.control_channel.toolbox_execute,
                    engine_id=engine_id,
                    tool_call=call.to_dict(),
                    timeout_seconds=float(timeout_seconds or 30.0),
                    tools_view=tools_view_payload,
                    callback_binding=dict(callback_binding or {}) or None,
                )
        except Exception as exc:
            if _is_coarse_cancel_execution_error(exc):
                call.error = f"Execution canceled: sandbox_recycled:{str(call.name or '').strip()}"
                return call
            raise
        finally:
            if 'callback_binding' in locals() and callback_binding and hasattr(self, "_callback_relay"):
                self._callback_relay.release_session(str(callback_binding.get("session_token") or ""))
        payload = dict(rpc_out or {})
        tool_out = dict(payload.get("tool_call") or {})
        return ToolCall.from_dict(tool_out) if tool_out else call

    async def _select_engine_id(self) -> str:
        if str(self.config.sandbox_toolbox_id or "").strip():
            return ""
        if self.control_channel is None:
            raise RuntimeError("control_channel_not_configured")
        engine_ids = [str(item or "").strip() for item in list(self.config.sandbox_engine_ids or []) if str(item or "").strip()]
        if not engine_ids:
            raise RuntimeError("sandbox_engine_ids_required")
        if len(engine_ids) == 1 or str(self.config.sandbox_selection or "round_robin").strip().lower() != "round_robin":
            return engine_ids[0]
        async with self._rr_lock:
            engine_id = engine_ids[self._rr_index % len(engine_ids)]
            self._rr_index = (self._rr_index + 1) % max(1, len(engine_ids))
            return engine_id


class HostedToolBoxRef:
    def __init__(
        self,
        *,
        toolbox_id: str,
        host: Any,
        python_executable: Optional[str] = None,
        worker_profile_class: str = "generic",
    ) -> None:
        self.toolbox_id = str(toolbox_id or "").strip()
        if not self.toolbox_id:
            raise ValueError("toolbox_id_required")
        self.host = host
        self.python_executable = str(python_executable or "").strip() or None
        self.worker_profile_class = str(worker_profile_class or "generic").strip() or "generic"

    @property
    def ref_name(self) -> str:
        return self.toolbox_id

    def _host_descriptor(self) -> Dict[str, Any]:
        host = self.host
        host_type = type(host).__name__
        descriptor: Dict[str, Any] = {
            "host_type": host_type,
        }
        if hasattr(host, "control_settings"):
            descriptor["kind"] = "control_channel"
            descriptor["control_settings"] = dict(getattr(host, "control_settings", {}) or {})
            return descriptor
        engines_state_file = getattr(host, "engines_state_file", None)
        control_state_file = getattr(host, "control_state_file", None)
        if engines_state_file is not None or control_state_file is not None:
            descriptor["kind"] = "service"
            descriptor["engines_state_file"] = str(engines_state_file) if engines_state_file is not None else None
            descriptor["control_state_file"] = str(control_state_file) if control_state_file is not None else None
            return descriptor
        descriptor["kind"] = "opaque"
        return descriptor

    def to_dict(self) -> Dict[str, Any]:
        return {
            "toolbox_id": self.toolbox_id,
            "python_executable": self.python_executable,
            "worker_profile_class": self.worker_profile_class,
            "host": self._host_descriptor(),
        }

    @classmethod
    def from_dict(
        cls,
        payload: Dict[str, Any],
        *,
        host: Any = None,
    ) -> "HostedToolBoxRef":
        row = dict(payload or {})
        resolved_host = host
        if resolved_host is None:
            host_row = dict(row.get("host") or {})
            kind = str(host_row.get("kind") or "").strip().lower()
            if kind == "control_channel":
                from .engine_host_channel import EngineHostControlChannel

                resolved_host = EngineHostControlChannel(dict(host_row.get("control_settings") or {}))
            elif kind == "service":
                from .engine_host_service import EngineHostService

                engines_state_raw = str(host_row.get("engines_state_file") or "").strip()
                control_state_raw = str(host_row.get("control_state_file") or "").strip()
                resolved_host = EngineHostService(
                    engines_state_file=Path(engines_state_raw) if engines_state_raw else None,
                    control_state_file=Path(control_state_raw) if control_state_raw else None,
                )
            else:
                raise ValueError("host_required_for_hosted_toolbox_ref_deserialization")
        return cls(
            toolbox_id=str(row.get("toolbox_id") or "").strip(),
            host=resolved_host,
            python_executable=str(row.get("python_executable") or "").strip() or None,
            worker_profile_class=str(row.get("worker_profile_class") or "generic").strip() or "generic",
        )

    def mutate(self) -> "PendingHostedToolboxRef":
        return PendingHostedToolboxRef(self)

    def register_auto_callable(
        self,
        *,
        relative_path: str,
        content: str,
        module_name: str,
        callable_name: str,
        environment_name: str = "base",
        required_imports: Optional[Sequence[str]] = None,
        sandbox_policy: Optional[Dict[str, Any]] = None,
        activate: bool = True,
        hidden: bool = False,
        non_restartable: bool = False,
        guide_content: Optional[Dict[str, List[str]]] = None,
        guide_description: Optional[str] = None,
        callback_signature: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        request = {
            "files": [
                ToolboxBundleFile(
                    relative_path=str(relative_path or "").strip(),
                    content=str(content or ""),
                ).to_runtime_dict()
            ],
            "module_name": str(module_name or "").strip(),
            "callable_name": str(callable_name or "").strip(),
            "sandbox_profile": SandboxProfileSpec(
                environment_name=str(environment_name or "base").strip() or "base",
                required_imports=[str(item or "").strip() for item in list(required_imports or []) if str(item or "").strip()],
                sandbox_policy=dict(sandbox_policy or {}),
            ).to_dict(),
            "activate": bool(activate),
            "hidden": bool(hidden),
            "non_restartable": bool(non_restartable),
            "guide_content": dict(guide_content or {}) or None,
            "guide_description": str(guide_description or "").strip() or None,
            "callback_signature": dict(callback_signature or {}) or None,
        }
        return dict(
            self.host.toolbox_register_auto(
                toolbox_id=self.toolbox_id,
                requests=[request],
                python_executable=self.python_executable,
                worker_profile_class=self.worker_profile_class,
            )
            or {}
        )

    def add_auto_callable(self, **kwargs: Any) -> Dict[str, Any]:
        return self.register_auto_callable(**kwargs)

    def register_python_callable(
        self,
        implementation: Any,
        *,
        environment_name: str = "base",
        required_imports: Optional[Sequence[str]] = None,
        sandbox_policy: Optional[Dict[str, Any]] = None,
        activate: bool = True,
        hidden: bool = False,
        non_restartable: bool = False,
        guide_content: Optional[Dict[str, List[str]]] = None,
        guide_description: Optional[str] = None,
        callback_signature: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        module = inspect.getmodule(implementation)
        module_name = str(getattr(implementation, "__module__", "") or getattr(module, "__name__", "") or "").strip()
        if not module_name:
            raise ValueError("callable_module_name_required")
        callable_name = str(getattr(implementation, "__name__", "") or "").strip()
        if not callable_name:
            raise ValueError("callable_name_required")
        source_path = inspect.getsourcefile(implementation) or getattr(module, "__file__", None)
        if not source_path:
            raise ValueError("callable_source_file_required")
        source_file = Path(str(source_path)).expanduser().resolve()
        if not source_file.exists():
            raise ValueError("callable_source_file_missing")
        return self.register_auto_callable(
            relative_path=source_file.name,
            content=source_file.read_text(encoding="utf-8"),
            module_name=module_name,
            callable_name=callable_name,
            environment_name=environment_name,
            required_imports=required_imports,
            sandbox_policy=sandbox_policy,
            activate=activate,
            hidden=hidden,
            non_restartable=non_restartable,
            guide_content=guide_content,
            guide_description=guide_description,
            callback_signature=callback_signature,
        )

    def add_python_callable(self, implementation: Any, **kwargs: Any) -> Dict[str, Any]:
        return self.register_python_callable(implementation, **kwargs)

    def register_manual_tool(
        self,
        tool_definition: Dict[str, Any],
        implementation: Any,
        *,
        environment_name: str = "base",
        required_imports: Optional[Sequence[str]] = None,
        sandbox_policy: Optional[Dict[str, Any]] = None,
        hidden: bool = False,
        non_restartable: bool = False,
        callback_signature: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        module = inspect.getmodule(implementation)
        module_name = str(getattr(implementation, "__module__", "") or getattr(module, "__name__", "") or "").strip()
        if not module_name:
            raise ValueError("callable_module_name_required")
        callable_name = str(getattr(implementation, "__name__", "") or "").strip()
        if not callable_name:
            raise ValueError("callable_name_required")
        source_path = inspect.getsourcefile(implementation) or getattr(module, "__file__", None)
        if not source_path:
            raise ValueError("callable_source_file_required")
        source_file = Path(str(source_path)).expanduser().resolve()
        if not source_file.exists():
            raise ValueError("callable_source_file_missing")
        return dict(
            self.host.toolbox_register_manual(
                toolbox_id=self.toolbox_id,
                requests=[
                    {
                        "files": [
                            ToolboxBundleFile(
                                relative_path=source_file.name,
                                content=source_file.read_text(encoding="utf-8"),
                            ).to_runtime_dict()
                        ],
                        "module_name": module_name,
                        "callable_name": callable_name,
                        "tool_definition": dict(tool_definition or {}),
                        "sandbox_profile": SandboxProfileSpec(
                            environment_name=str(environment_name or "base").strip() or "base",
                            required_imports=[str(item or "").strip() for item in list(required_imports or []) if str(item or "").strip()],
                            sandbox_policy=dict(sandbox_policy or {}),
                        ).to_dict(),
                        "hidden": bool(hidden),
                        "non_restartable": bool(non_restartable),
                        "callback_signature": dict(callback_signature or {}) or None,
                    }
                ],
                python_executable=self.python_executable,
                worker_profile_class=self.worker_profile_class,
            )
            or {}
        )

    def add_manual_tool(self, tool_definition: Dict[str, Any], implementation: Any, **kwargs: Any) -> Dict[str, Any]:
        return self.register_manual_tool(tool_definition, implementation, **kwargs)

    def unregister_manual_tool(self, *, module_name: str, callable_name: str) -> Dict[str, Any]:
        key = f"manual:{str(module_name or '').strip()}:{str(callable_name or '').strip()}"
        return dict(
            self.host.toolbox_unregister_manual(
                toolbox_id=self.toolbox_id,
                tool_keys=[key],
                python_executable=self.python_executable,
                worker_profile_class=self.worker_profile_class,
            )
            or {}
        )

    def remove_manual_tool(self, *, module_name: str, callable_name: str) -> Dict[str, Any]:
        return self.unregister_manual_tool(module_name=module_name, callable_name=callable_name)

    def unregister_auto_callable(self, *, module_name: str, callable_name: str) -> Dict[str, Any]:
        key = f"{str(module_name or '').strip()}:{str(callable_name or '').strip()}"
        return dict(
            self.host.toolbox_unregister_auto(
                toolbox_id=self.toolbox_id,
                tool_keys=[key],
                python_executable=self.python_executable,
                worker_profile_class=self.worker_profile_class,
            )
            or {}
        )

    def remove_auto_callable(self, *, module_name: str, callable_name: str) -> Dict[str, Any]:
        return self.unregister_auto_callable(module_name=module_name, callable_name=callable_name)

    def register_intrinsic_tools(
        self,
        intrinsic_tool_names: Sequence[str],
        *,
        include_guides: bool = False,
        environment_name: str = "base",
        sandbox_policy: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return dict(
            self.host.toolbox_register_intrinsics(
                toolbox_id=self.toolbox_id,
                intrinsic_tool_names=[str(item or "").strip() for item in list(intrinsic_tool_names or []) if str(item or "").strip()],
                include_guides=bool(include_guides),
                sandbox_profile=SandboxProfileSpec(
                    environment_name=str(environment_name or "base").strip() or "base",
                    sandbox_policy=dict(sandbox_policy or {}),
                ).to_dict(),
                python_executable=self.python_executable,
                worker_profile_class=self.worker_profile_class,
            )
            or {}
        )

    def add_intrinsic_tools(self, intrinsic_tool_names: Sequence[str], **kwargs: Any) -> Dict[str, Any]:
        return self.register_intrinsic_tools(intrinsic_tool_names, **kwargs)

    def environment_descriptions(self) -> Dict[str, Any]:
        return dict(self.host.toolbox_environment_description_list() or {})

    def list_environment_descriptions(self) -> Dict[str, Any]:
        return self.environment_descriptions()

    def upsert_environment_description(
        self,
        *,
        name: str,
        base_env_name: Optional[str] = None,
        extra_packages: Optional[Sequence[str]] = None,
        allow_online_install: bool = False,
    ) -> Dict[str, Any]:
        return dict(
            self.host.toolbox_environment_description_upsert(
                name=str(name or "").strip(),
                base_env_name=str(base_env_name or "").strip() or None,
                extra_packages=[str(item or "").strip() for item in list(extra_packages or []) if str(item or "").strip()],
                allow_online_install=bool(allow_online_install),
            )
            or {}
        )

    def clone_environment_description(
        self,
        *,
        source_name: str,
        target_name: str,
        extra_packages: Optional[Sequence[str]] = None,
        allow_online_install: Optional[bool] = None,
    ) -> Dict[str, Any]:
        return dict(
            self.host.toolbox_environment_description_clone(
                source_name=str(source_name or "").strip(),
                target_name=str(target_name or "").strip(),
                extra_packages=[str(item or "").strip() for item in list(extra_packages or []) if str(item or "").strip()] if extra_packages is not None else None,
                allow_online_install=allow_online_install,
            )
            or {}
        )

    def resolve_environment_requirements(
        self,
        *,
        environment_name: str,
        tool_keys: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        return dict(
            self.host.toolbox_environment_resolve_requirements(
                toolbox_id=self.toolbox_id,
                environment_name=str(environment_name or "base").strip() or "base",
                tool_keys=[str(item or "").strip() for item in list(tool_keys or []) if str(item or "").strip()] or None,
            )
            or {}
        )

    def apply_environment_description(
        self,
        *,
        environment_name: str,
        toolbox_ids: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        return dict(
            self.host.toolbox_environment_apply(
                environment_name=str(environment_name or "base").strip() or "base",
                toolbox_ids=[str(item or "").strip() for item in list(toolbox_ids or []) if str(item or "").strip()] or None,
            )
            or {}
        )

    def realize_environment(
        self,
        *,
        environment_name: str,
        tool_keys: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        return dict(
            self.host.toolbox_environment_realize(
                toolbox_id=self.toolbox_id,
                environment_name=str(environment_name or "base").strip() or "base",
                tool_keys=[str(item or "").strip() for item in list(tool_keys or []) if str(item or "").strip()] or None,
            )
            or {}
        )

    def sync_environment_description(
        self,
        *,
        source_environment_name: str,
        target_environment_name: Optional[str] = None,
        tool_keys: Optional[Sequence[str]] = None,
        apply: bool = False,
        realize: bool = False,
    ) -> Dict[str, Any]:
        return dict(
            self.host.toolbox_environment_sync_description(
                toolbox_id=self.toolbox_id,
                source_environment_name=str(source_environment_name or "base").strip() or "base",
                target_environment_name=str(target_environment_name or "").strip() or None,
                tool_keys=[str(item or "").strip() for item in list(tool_keys or []) if str(item or "").strip()] or None,
                apply=bool(apply),
                realize=bool(realize),
            )
            or {}
        )

    def prepare_environment_install(
        self,
        *,
        environment_name: str,
        tool_keys: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        return dict(
            self.host.toolbox_environment_prepare_install(
                toolbox_id=self.toolbox_id,
                environment_name=str(environment_name or "base").strip() or "base",
                tool_keys=[str(item or "").strip() for item in list(tool_keys or []) if str(item or "").strip()] or None,
            )
            or {}
        )

    def lock_environment_install(
        self,
        *,
        environment_name: str,
        tool_keys: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        return dict(
            self.host.toolbox_environment_lock_install(
                toolbox_id=self.toolbox_id,
                environment_name=str(environment_name or "base").strip() or "base",
                tool_keys=[str(item or "").strip() for item in list(tool_keys or []) if str(item or "").strip()] or None,
            )
            or {}
        )

    def resolve_environment_install_lock(
        self,
        *,
        environment_name: str,
        tool_keys: Optional[Sequence[str]] = None,
        allow_resolution: bool = False,
    ) -> Dict[str, Any]:
        return dict(
            self.host.toolbox_environment_resolve_install_lock(
                toolbox_id=self.toolbox_id,
                environment_name=str(environment_name or "base").strip() or "base",
                tool_keys=[str(item or "").strip() for item in list(tool_keys or []) if str(item or "").strip()] or None,
                allow_resolution=bool(allow_resolution),
            )
            or {}
        )

    def verify_environment_install_lock(
        self,
        *,
        environment_name: str,
        tool_keys: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        return dict(
            self.host.toolbox_environment_verify_install_lock(
                toolbox_id=self.toolbox_id,
                environment_name=str(environment_name or "base").strip() or "base",
                tool_keys=[str(item or "").strip() for item in list(tool_keys or []) if str(item or "").strip()] or None,
            )
            or {}
        )

    def verify_environment_install_receipt(
        self,
        *,
        environment_name: str,
        tool_keys: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        return dict(
            self.host.toolbox_environment_verify_install_receipt(
                toolbox_id=self.toolbox_id,
                environment_name=str(environment_name or "base").strip() or "base",
                tool_keys=[str(item or "").strip() for item in list(tool_keys or []) if str(item or "").strip()] or None,
            )
            or {}
        )

    def execute_environment_install(
        self,
        *,
        environment_name: str,
        tool_keys: Optional[Sequence[str]] = None,
        allow_execution: bool = False,
    ) -> Dict[str, Any]:
        return dict(
            self.host.toolbox_environment_execute_install(
                toolbox_id=self.toolbox_id,
                environment_name=str(environment_name or "base").strip() or "base",
                tool_keys=[str(item or "").strip() for item in list(tool_keys or []) if str(item or "").strip()] or None,
                allow_execution=bool(allow_execution),
            )
            or {}
        )

    def unregister_intrinsic_tools(
        self,
        intrinsic_tool_names: Sequence[str],
        *,
        include_guides: bool = False,
    ) -> Dict[str, Any]:
        return dict(
            self.host.toolbox_unregister_intrinsics(
                toolbox_id=self.toolbox_id,
                intrinsic_tool_names=[str(item or "").strip() for item in list(intrinsic_tool_names or []) if str(item or "").strip()],
                include_guides=bool(include_guides),
                python_executable=self.python_executable,
                worker_profile_class=self.worker_profile_class,
            )
            or {}
        )

    def remove_intrinsic_tools(
        self,
        intrinsic_tool_names: Sequence[str],
        *,
        include_guides: bool = False,
    ) -> Dict[str, Any]:
        return self.unregister_intrinsic_tools(intrinsic_tool_names, include_guides=include_guides)

    def describe(self, *, timeout_seconds: float = 10.0) -> Dict[str, Any]:
        return dict(
            self.host.toolbox_describe(
                toolbox_id=self.toolbox_id,
                timeout_seconds=float(timeout_seconds or 10.0),
            )
            or {}
        )

    def gate(self, *, tool_name: str, tools_view: Optional[ToolsView] = None) -> Dict[str, Any]:
        return dict(
            self.host.toolbox_gate(
                toolbox_id=self.toolbox_id,
                tool_name=str(tool_name or "").strip(),
                tools_view=serialize_tools_view(tools_view),
            )
            or {}
        )

    def list_tools(self, *, timeout_seconds: float = 10.0) -> Dict[str, Any]:
        return self.describe(timeout_seconds=timeout_seconds)

    def execute(
        self,
        *,
        tool_name: str,
        arguments: Optional[Dict[str, Any]] = None,
        timeout_seconds: float = 30.0,
        tools_view: Optional[ToolsView] = None,
        callback_processor: Optional[Callable[..., Any]] = None,
        callback_context: Any = None,
        tool_call_id: str = "",
    ) -> Dict[str, Any]:
        name = str(tool_name or "").strip()
        if not name:
            raise ValueError("tool_name_required")
        call_id = str(tool_call_id or "").strip() or secrets.token_hex(12)
        callback_binding = None
        if callable(callback_processor):
            if not hasattr(self, "_callback_relay"):
                self._callback_relay = _HostedToolCallbackRelay()
            signature = None
            try:
                tool_meta = dict(self.describe().get("tool_metadata") or {}).get(name) or {}
                signature = dict(tool_meta.get("callback_signature") or {}) or None
            except Exception:
                signature = None
            callback_binding = self._callback_relay.bind_session(
                processor=callback_processor,
                toolbox_id=self.toolbox_id,
                tool_name=name,
                tool_call_id=call_id,
                tool_arguments=dict(arguments or {}),
                callback_signature=signature,
                user_context=callback_context,
            )
        try:
            return dict(
                self.host.toolbox_execute(
                    toolbox_id=self.toolbox_id,
                    tool_call={
                        "id": call_id,
                        "name": name,
                        "arguments": dict(arguments or {}),
                    },
                    timeout_seconds=float(timeout_seconds or 30.0),
                    tools_view=serialize_tools_view(tools_view),
                    callback_binding=dict(callback_binding or {}) or None,
                )
                or {}
            )
        finally:
            if callback_binding and hasattr(self, "_callback_relay"):
                self._callback_relay.release_session(str(callback_binding.get("session_token") or ""))

    def cancel(
        self,
        *,
        tool_name: str = "",
        tool_call_id: str = "",
        timeout_seconds: float = 8.0,
        respawn: bool = True,
    ) -> Dict[str, Any]:
        return dict(
            self.host.toolbox_cancel(
                toolbox_id=self.toolbox_id,
                tool_name=str(tool_name or "").strip(),
                tool_call_id=str(tool_call_id or "").strip(),
                timeout_seconds=float(timeout_seconds or 8.0),
                respawn=bool(respawn),
            )
            or {}
        )



class PendingHostedToolboxRef:
    def __init__(self, base_ref: HostedToolBoxRef) -> None:
        self.base_ref = base_ref
        self._pending_auto_requests: List[Dict[str, Any]] = []
        self._pending_manual_requests: List[Dict[str, Any]] = []

    def register_auto_callable(
        self,
        *,
        relative_path: str,
        content: str,
        module_name: str,
        callable_name: str,
        environment_name: str = "base",
        required_imports: Optional[Sequence[str]] = None,
        sandbox_policy: Optional[Dict[str, Any]] = None,
        activate: bool = True,
        hidden: bool = False,
        non_restartable: bool = False,
        guide_content: Optional[Dict[str, List[str]]] = None,
        guide_description: Optional[str] = None,
        callback_signature: Optional[Dict[str, Any]] = None,
    ) -> "PendingHostedToolboxRef":
        request = {
            "files": [
                ToolboxBundleFile(
                    relative_path=str(relative_path or "").strip(),
                    content=str(content or ""),
                ).to_runtime_dict()
            ],
            "module_name": str(module_name or "").strip(),
            "callable_name": str(callable_name or "").strip(),
            "sandbox_profile": SandboxProfileSpec(
                environment_name=str(environment_name or "base").strip() or "base",
                required_imports=[str(item or "").strip() for item in list(required_imports or []) if str(item or "").strip()],
                sandbox_policy=dict(sandbox_policy or {}),
            ).to_dict(),
            "activate": bool(activate),
            "hidden": bool(hidden),
            "non_restartable": bool(non_restartable),
            "guide_content": dict(guide_content or {}) or None,
            "guide_description": str(guide_description or "").strip() or None,
            "callback_signature": dict(callback_signature or {}) or None,
        }
        self._pending_auto_requests.append(request)
        return self

    def add_auto_callable(self, **kwargs: Any) -> "PendingHostedToolboxRef":
        return self.register_auto_callable(**kwargs)

    def register_python_callable(
        self,
        implementation: Any,
        *,
        environment_name: str = "base",
        required_imports: Optional[Sequence[str]] = None,
        sandbox_policy: Optional[Dict[str, Any]] = None,
        activate: bool = True,
        hidden: bool = False,
        non_restartable: bool = False,
        guide_content: Optional[Dict[str, List[str]]] = None,
        guide_description: Optional[str] = None,
        callback_signature: Optional[Dict[str, Any]] = None,
    ) -> "PendingHostedToolboxRef":
        module = inspect.getmodule(implementation)
        module_name = str(getattr(implementation, "__module__", "") or getattr(module, "__name__", "") or "").strip()
        if not module_name:
            raise ValueError("callable_module_name_required")
        callable_name = str(getattr(implementation, "__name__", "") or "").strip()
        if not callable_name:
            raise ValueError("callable_name_required")
        source_path = inspect.getsourcefile(implementation) or getattr(module, "__file__", None)
        if not source_path:
            raise ValueError("callable_source_file_required")
        source_file = Path(str(source_path)).expanduser().resolve()
        if not source_file.exists():
            raise ValueError("callable_source_file_missing")
        return self.register_auto_callable(
            relative_path=source_file.name,
            content=source_file.read_text(encoding="utf-8"),
            module_name=module_name,
            callable_name=callable_name,
            environment_name=environment_name,
            required_imports=required_imports,
            sandbox_policy=sandbox_policy,
            activate=activate,
            hidden=hidden,
            non_restartable=non_restartable,
            guide_content=guide_content,
            guide_description=guide_description,
            callback_signature=callback_signature,
        )

    def add_python_callable(self, implementation: Any, **kwargs: Any) -> "PendingHostedToolboxRef":
        return self.register_python_callable(implementation, **kwargs)

    def register_manual_tool(
        self,
        tool_definition: Dict[str, Any],
        implementation: Any,
        *,
        environment_name: str = "base",
        required_imports: Optional[Sequence[str]] = None,
        sandbox_policy: Optional[Dict[str, Any]] = None,
        hidden: bool = False,
        non_restartable: bool = False,
        callback_signature: Optional[Dict[str, Any]] = None,
    ) -> "PendingHostedToolboxRef":
        module = inspect.getmodule(implementation)
        module_name = str(getattr(implementation, "__module__", "") or getattr(module, "__name__", "") or "").strip()
        if not module_name:
            raise ValueError("callable_module_name_required")
        callable_name = str(getattr(implementation, "__name__", "") or "").strip()
        if not callable_name:
            raise ValueError("callable_name_required")
        source_path = inspect.getsourcefile(implementation) or getattr(module, "__file__", None)
        if not source_path:
            raise ValueError("callable_source_file_required")
        source_file = Path(str(source_path)).expanduser().resolve()
        if not source_file.exists():
            raise ValueError("callable_source_file_missing")
        
        request = {
            "files": [
                ToolboxBundleFile(
                    relative_path=source_file.name,
                    content=source_file.read_text(encoding="utf-8"),
                ).to_runtime_dict()
            ],
            "module_name": module_name,
            "callable_name": callable_name,
            "tool_definition": dict(tool_definition or {}),
            "sandbox_profile": SandboxProfileSpec(
                environment_name=str(environment_name or "base").strip() or "base",
                required_imports=[str(item or "").strip() for item in list(required_imports or []) if str(item or "").strip()],
                sandbox_policy=dict(sandbox_policy or {}),
            ).to_dict(),
            "hidden": bool(hidden),
            "non_restartable": bool(non_restartable),
            "callback_signature": dict(callback_signature or {}) or None,
        }
        self._pending_manual_requests.append(request)
        return self

    def add_manual_tool(self, tool_definition: Dict[str, Any], implementation: Any, **kwargs: Any) -> "PendingHostedToolboxRef":
        return self.register_manual_tool(tool_definition, implementation, **kwargs)

    def resolve_sandbox(self) -> HostedToolBoxRef:
        if self._pending_auto_requests:
            self.base_ref.host.toolbox_register_auto(
                toolbox_id=self.base_ref.toolbox_id,
                requests=list(self._pending_auto_requests),
                python_executable=self.base_ref.python_executable,
                worker_profile_class=self.base_ref.worker_profile_class,
            )
        if self._pending_manual_requests:
            self.base_ref.host.toolbox_register_manual(
                toolbox_id=self.base_ref.toolbox_id,
                requests=list(self._pending_manual_requests),
                python_executable=self.base_ref.python_executable,
                worker_profile_class=self.base_ref.worker_profile_class,
            )
        self._pending_auto_requests.clear()
        self._pending_manual_requests.clear()
        return self.base_ref

SandboxedToolboxFacade = HostedToolBoxRef


def load_toolbox_from_manifest(manifest_path: Path) -> tuple[Toolbox, Dict[str, Any]]:
    manifest_file = Path(manifest_path).expanduser().resolve()
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("toolbox_manifest_invalid")
    bundle_root = manifest_file.parent
    files_root = (bundle_root / "files").resolve()
    if str(files_root) not in sys.path:
        sys.path.insert(0, str(files_root))
    intrinsic_tool_names = [
        str(item or "").strip()
        for item in list(manifest.get("intrinsic_tool_names") or [])
        if str(item or "").strip()
    ]
    toolbox = Toolbox()
    hidden_user_tools = [
        str(item or "").strip()
        for item in list(manifest.get("hidden_tool_names") or [])
        if str(item or "").strip()
    ]
    if intrinsic_tool_names:
        ok, msg = toolbox.add_tool_callable(
            intrinsic_tool_names,
            is_intrinsic=True,
            include_guides=bool(manifest.get("with_intrinsic_guides", False)),
            activate=True,
        )
        if not ok:
            raise ValueError(str(msg or "intrinsic_registration_failed"))
        active_intrinsic = [
            str(item or "").strip()
            for item in list(manifest.get("active_intrinsic_tool_names") or [])
            if str(item or "").strip()
        ]
        hidden_intrinsic = [
            str(item or "").strip()
            for item in list(manifest.get("hidden_intrinsic_tool_names") or [])
            if str(item or "").strip()
        ]
        if active_intrinsic:
            toolbox.active_intrinsic_tool_names = [
                name for name in active_intrinsic if name in toolbox.intrinsic_tools
            ]
        if hidden_intrinsic:
            toolbox.hidden_intrinsic_tool_names = [
                name for name in hidden_intrinsic if name in toolbox.intrinsic_tools
            ]
    for item in list(manifest.get("auto_tools") or []):
        auto_meta = dict(item or {})
        module_name = str(auto_meta.get("module_name") or "").strip()
        callable_name = str(auto_meta.get("callable_name") or "").strip()
        if not module_name:
            raise ValueError("auto_tool_module_name_required")
        if not callable_name:
            raise ValueError("auto_tool_callable_name_required")
        module = importlib.import_module(module_name)
        ok, msg = toolbox.add_tool_callable(
            callable_name,
            search_scope=dict(vars(module)),
            activate=bool(auto_meta.get("activate", True)),
            guide_content=dict(auto_meta.get("guide_content") or {}) or None,
            guide_description=str(auto_meta.get("guide_description") or "").strip() or None,
        )
        if not ok:
            raise ValueError(str(msg or "auto_tool_registration_failed"))
        tool_def = toolbox.get_tool(callable_name)
        if tool_def is not None:
            tool_def["callback_signature"] = dict(auto_meta.get("callback_signature") or {}) or None
    for item in list(manifest.get("tools") or []):
        tool_meta = dict(item or {})
        entrypoint = str(tool_meta.get("entrypoint") or "").strip()
        if ":" not in entrypoint:
            raise ValueError(f"tool_entrypoint_invalid:{entrypoint}")
        module_name, attr_name = entrypoint.split(":", 1)
        module = importlib.import_module(module_name)
        implementation = getattr(module, attr_name)
        ok, msg = toolbox.add_tool_external(
            tool_definition=dict(tool_meta.get("definition") or {}),
            implementation=implementation,
            activate=True,
            allow_override=False,
        )
        if not ok:
            raise ValueError(str(msg or "tool_registration_failed"))
        tool_name = str(dict(tool_meta.get("definition") or {}).get("function", {}).get("name") or "").strip()
        tool_def = toolbox.get_tool(tool_name) if tool_name else None
        if tool_def is not None:
            tool_def["callback_signature"] = dict(tool_meta.get("callback_signature") or {}) or None
    if hidden_user_tools:
        toolbox.hidden_tool_names = [
            name for name in hidden_user_tools if name in toolbox.tools
        ]
    return toolbox, manifest
