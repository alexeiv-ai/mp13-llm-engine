from __future__ import annotations

import argparse
import asyncio
import json
import os
import queue
import socket
import sys
import threading
import traceback
from multiprocessing.connection import Client, Listener
from pathlib import Path
from typing import Any, Dict, Optional

from mp13_engine.mp13_config import ToolCall

from .callable_surface import (
    HOST_CAPABILITY_APPROVAL_CALLBACK_NAME,
    HOST_CAPABILITY_DISPATCH_CALLBACK_NAME,
    host_capability_approval_request,
    toolbox_brokered_io_call_surface,
)
from .sandbox.host_capabilities import HostCapabilityBroker, HostCapabilityProviderCall
from .sandbox.service_broker_registry import (
    invoke_service_broker_method,
    service_broker_host_capability_session,
)
from .toolbox_harness import ToolboxWorkerStartupSpec, load_toolbox_from_manifest

PROTOCOL_VERSION = 1


def _contract_name() -> str:
    return str(os.environ.get("MP13_WORKER_CONTRACT") or "mp13.toolbox.rpc.v1").strip() or "mp13.toolbox.rpc.v1"


_toolbox_lock = threading.Lock()
_toolbox_state: Optional[tuple[Any, Dict[str, Any]]] = None
_startup_spec_lock = threading.Lock()
_startup_spec: Optional[ToolboxWorkerStartupSpec] = None


def _worker_engine_id() -> str:
    spec = _startup_spec_or_none()
    if spec and str(spec.worker_id or "").strip():
        return str(spec.worker_id).strip()
    return str(os.environ.get("MP13_TOOLBOX_EXECUTOR_ENGINE_ID") or os.environ.get("MP13_ENGINE_ID") or "toolbox").strip() or "toolbox"


def _startup_spec_or_none() -> Optional[ToolboxWorkerStartupSpec]:
    global _startup_spec
    with _startup_spec_lock:
        if _startup_spec is not None:
            return _startup_spec
        raw = str(os.environ.get("MP13_TOOLBOX_WORKER_SPEC_PATH") or "").strip()
        if not raw:
            return None
        path = Path(raw).expanduser().resolve()
        payload = json.loads(path.read_text(encoding="utf-8"))
        _startup_spec = ToolboxWorkerStartupSpec.from_dict(dict(payload or {}))
        return _startup_spec


def _host_service():
    from .service.host_service import EngineHostService

    spec = _startup_spec_or_none()
    engines_state_file = (
        str(spec.engines_state_file or "").strip()
        if spec is not None
        else str(os.environ.get("MP13_HOSTING_ENGINES_STATE_FILE") or "").strip()
    )
    control_state_file = (
        str(spec.control_state_file or "").strip()
        if spec is not None
        else str(os.environ.get("MP13_HOSTING_CONTROL_STATE_FILE") or "").strip()
    )
    svc = EngineHostService(
        engines_state_file=Path(engines_state_file).expanduser().resolve() if engines_state_file else None,
        control_state_file=Path(control_state_file).expanduser().resolve() if control_state_file else None,
    )
    if spec is not None and str(spec.worker_id or "").strip() and callable(getattr(svc, "register_spawned", None)):
        svc.register_spawned(
            engine_id=str(spec.worker_id).strip(),
            pid=os.getpid(),
            command=[sys.executable, "-m", "hosting.toolbox_executor_ipc"],
            env={
                "MP13_TOOLBOX_EXECUTOR_ENGINE_ID": str(spec.worker_id).strip(),
                "MP13_TOOLBOX_WORKER_SPEC_PATH": str(os.environ.get("MP13_TOOLBOX_WORKER_SPEC_PATH") or "").strip(),
            },
            executor_kind="toolbox_executor",
            sandbox_policy=dict(spec.policy or {}),
        )
    return svc


def _toolbox_service_broker_provider_invoker(
    svc: Any,
    *,
    callback_context: Optional[Dict[str, Any]] = None,
):
    def _invoke(session: Any, call: HostCapabilityProviderCall) -> Dict[str, Any]:
        engine_id = str(call.context.engine_id or dict(getattr(session, "binding", {}) or {}).get("engine_id") or "").strip()
        result = invoke_service_broker_method(
            svc,
            engine_id=engine_id,
            method=call.method,
            arguments=dict(call.arguments or {}),
            callback_context=callback_context,
        )
        return {
            "status": "ok",
            "provider_call_id": call.provider_call_id,
            "result": dict(result or {}),
        }

    return _invoke


def _toolbox_approval_requester(
    callback_binding: Optional[Dict[str, Any]],
    *,
    callback_context: Optional[Dict[str, Any]] = None,
):
    binding = dict(callback_binding or {})
    if not binding:
        return None

    def _request_approval(payload: Dict[str, Any]) -> Dict[str, Any]:
        response = _invoke_callback_binding(
            binding,
            callback_name=HOST_CAPABILITY_APPROVAL_CALLBACK_NAME,
            payload=host_capability_approval_request(dict(payload or {})),
            context=dict(callback_context or {}),
        )
        return dict(response.get("result") or response or {})

    return _request_approval


def _toolbox_host_capability_broker(
    *,
    engine_id: str,
    approval: Optional[Dict[str, Any]] = None,
    callback_binding: Optional[Dict[str, Any]] = None,
    callback_context: Optional[Dict[str, Any]] = None,
    svc: Any = None,
) -> HostCapabilityBroker:
    eid = str(engine_id or "").strip() or _worker_engine_id()
    spec = _startup_spec_or_none()
    broker = HostCapabilityBroker(
        request_id=str(dict(callback_context or {}).get("tool_call_id") or ""),
        workflow_id=str(dict(callback_context or {}).get("workflow_id") or ""),
        package_id=str(dict(callback_context or {}).get("package_id") or ""),
        instance_id=str(dict(callback_context or {}).get("instance_id") or ""),
        engine_id=eid,
        consumer_id=eid,
        runtime_kind="toolbox_worker",
        policy=dict(spec.policy or {}) if spec is not None else {},
        provider_invoker=_toolbox_service_broker_provider_invoker(svc, callback_context=callback_context) if svc is not None else None,
        approval_requester=_toolbox_approval_requester(callback_binding, callback_context=callback_context),
        audit_emitter=getattr(svc, "_append_host_capability_audit_event", None) if svc is not None else None,
    )
    session = service_broker_host_capability_session(
        session_id=f"{eid}.service_broker",
        owner="service",
        visibility="consumer",
        scope={"consumer_id": eid},
        approval=dict(approval or {}),
        binding={"engine_id": eid},
    )
    broker.register_session(session)
    return broker


def _dispatch_host_capability_sync(broker: HostCapabilityBroker, call: Dict[str, Any]) -> Dict[str, Any]:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return broker.dispatch(call)
    result_queue: queue.Queue[tuple[str, Any]] = queue.Queue(maxsize=1)

    def _run() -> None:
        try:
            result_queue.put(("ok", asyncio.run(broker.dispatch_async(call))))
        except Exception as exc:
            result_queue.put(("error", exc))

    thread = threading.Thread(target=_run, name="toolbox-host-capability-dispatch", daemon=True)
    thread.start()
    status, payload = result_queue.get()
    thread.join(timeout=0.1)
    if status == "error":
        raise payload
    return dict(payload or {})


def _invoke_host_call(
    method: str,
    arguments: Dict[str, Any],
    *,
    callback_binding: Optional[Dict[str, Any]] = None,
    approval: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    req = dict(arguments or {})
    engine_id = str(req.get("engine_id") or _worker_engine_id()).strip() or _worker_engine_id()
    meth = str(method or "").strip()
    binding = dict(callback_binding or {})
    if not binding:
        raise RuntimeError("host_capability_dispatch_binding_missing")
    callback_context = dict(req.get("callback_context") or {}) if isinstance(req.get("callback_context"), dict) else {}
    response = _invoke_callback_binding(
        binding,
        callback_name=HOST_CAPABILITY_DISPATCH_CALLBACK_NAME,
        payload={
            "contract": "hosting.toolbox.host_capability_dispatch.v1",
            "method": meth,
            "arguments": req,
            "approval": dict(approval or {}),
        },
        context={
            "engine_id": engine_id,
            **dict(callback_context or {}),
        },
    )
    result = response.get("result")
    if isinstance(result, dict):
        if str(result.get("status") or "").strip().lower() == "error":
            raise RuntimeError(str(result.get("message") or result.get("reason") or "host_capability_dispatch_failed"))
        if "result" in result and len(result) <= 3:
            nested = result.get("result")
            return dict(nested or {}) if isinstance(nested, dict) else {"result": nested}
        return dict(result or {})
    return {"result": result}


def _invoke_callback_binding(
    binding: Dict[str, Any],
    *,
    callback_name: str,
    payload: Any,
    context: Dict[str, Any],
) -> Dict[str, Any]:
    family = str(binding.get("family") or "").strip() or ("AF_PIPE" if os.name == "nt" else "AF_UNIX")
    raw_address = binding.get("address")
    address = str(raw_address or "").strip()
    session_token = str(binding.get("session_token") or "").strip()
    if not address or not session_token:
        raise RuntimeError("callback_binding_invalid")
    try:
        conn = Client(address=address, family=family)
    except Exception as exc:
        token_prefix = session_token[:12]
        raise RuntimeError(
            f"callback_connect_failed:{family}:{address}:{token_prefix}:{type(exc).__name__}:{exc}"
        ) from exc
    try:
        conn.send(
            {
                "contract": str(binding.get("contract") or "hosting.toolbox.callbacks.v2"),
                "session_token": session_token,
                "callback_name": str(callback_name or "").strip(),
                "payload": payload,
                "context": dict(context or {}),
            }
        )
        try:
            response = dict(conn.recv() or {})
        except Exception as exc:
            token_prefix = session_token[:12]
            raise RuntimeError(
                f"callback_receive_failed:{family}:{address}:{token_prefix}:{type(exc).__name__}:{exc}"
            ) from exc
    finally:
        conn.close()
    if str(response.get("status") or "").strip().lower() == "error":
        raise RuntimeError(str(response.get("message") or "callback_invoke_failed"))
    return dict(response or {})


class HostCallbackClient:
    def __init__(
        self,
        *,
        engine_id: str,
        callback_binding: Optional[Dict[str, Any]] = None,
        toolbox_id: str = "",
        tool_name: str = "",
        tool_call_id: str = "",
        tool_arguments: Optional[Dict[str, Any]] = None,
        callback_signature: Optional[Dict[str, Any]] = None,
        host_api_approval: Optional[Dict[str, Any]] = None,
        user_context: Any = None,
    ) -> None:
        self.engine_id = str(engine_id or "").strip() or _worker_engine_id()
        self.callback_binding = dict(callback_binding or {})
        self.toolbox_id = str(toolbox_id or "").strip()
        self.tool_name = str(tool_name or "").strip()
        self.tool_call_id = str(tool_call_id or "").strip()
        self.tool_arguments = dict(tool_arguments or {})
        self.callback_signature = dict(callback_signature or {}) or None
        self.host_api_approval = dict(host_api_approval or {})
        self.user_context = user_context

    def _brokered_io_policy(self) -> Dict[str, Any]:
        spec = _startup_spec_or_none()
        if spec is not None and isinstance(spec.policy, dict):
            return dict(spec.policy or {})
        return {"sandbox": {"brokered_io": {"filesystem": True, "http": True, "subprocess": False}}}

    def _callback_context(self, *, method: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        context = {
            "engine_id": self.engine_id,
            "toolbox_id": self.toolbox_id,
            "tool_name": self.tool_name,
            "tool_call_id": self.tool_call_id or None,
            "tool_arguments": dict(self.tool_arguments or {}),
            "callback_signature": dict(self.callback_signature or {}) or None,
            "user_context": self.user_context,
        }
        policy = self._brokered_io_policy()
        context["callable_surface"] = toolbox_brokered_io_call_surface(
            str(method or ""),
            arguments=dict(arguments or {}),
            context=context,
            toolbox_policy=policy,
            host_capability_policy=policy,
            bridge_policy=policy,
            provider_id=self.engine_id,
            toolbox_id=self.toolbox_id,
            session_id=self.tool_call_id,
        )
        return context

    def call(self, method: str, arguments: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        meth = str(method or "").strip()
        req = dict(arguments or {})
        req.setdefault("engine_id", self.engine_id)
        req.setdefault("callback_context", self._callback_context(method=meth, arguments=req))
        if meth == "callback.invoke":
            callback_name = str(req.get("callback_name") or req.get("name") or "").strip()
            if not callback_name:
                raise RuntimeError("callback_name_required")
            if not self.callback_binding:
                raise RuntimeError("callback_binding_missing")
            response = _invoke_callback_binding(
                self.callback_binding,
                callback_name=callback_name,
                payload=req.get("payload"),
                context={
                    "engine_id": self.engine_id,
                    "toolbox_id": self.toolbox_id,
                    "tool_name": self.tool_name,
                    "tool_call_id": self.tool_call_id or None,
                    "tool_arguments": dict(self.tool_arguments or {}),
                    "callback_signature": dict(self.callback_signature or {}) or None,
                    "callable_surface": dict(dict(req.get("callback_context") or {}).get("callable_surface") or {}),
                },
            )
            return {"status": "ok", "callback_name": callback_name, "result": response.get("result")}
        approval = dict(req.pop("approval", None) or self.host_api_approval or {})
        return _invoke_host_call(meth, req, callback_binding=self.callback_binding, approval=approval)

    def describe(self) -> Dict[str, Any]:
        return _toolbox_host_capability_broker(
            engine_id=self.engine_id,
            approval=self.host_api_approval,
            callback_binding=self.callback_binding,
            callback_context=self._callback_context(method="host.describe", arguments={}),
        ).describe()


class BrokeredFsClient:
    def __init__(self, *, host: HostCallbackClient) -> None:
        self.host = host

    def list_dir(self, *, root_id: str, relative_path: Optional[str] = None) -> Dict[str, Any]:
        return self.host.call("fs.list", {"root_id": str(root_id or ""), "relative_path": relative_path})

    def read_text(self, *, root_id: str, relative_path: str, encoding: str = "utf-8") -> Dict[str, Any]:
        return self.host.call(
            "fs.read_text",
            {"root_id": str(root_id or ""), "relative_path": str(relative_path or ""), "encoding": str(encoding or "utf-8")},
        )

    def write_text(
        self,
        *,
        root_id: str,
        relative_path: str,
        text: str,
        encoding: str = "utf-8",
        create_parents: bool = True,
    ) -> Dict[str, Any]:
        return self.host.call(
            "fs.write_text",
            {
                "root_id": str(root_id or ""),
                "relative_path": str(relative_path or ""),
                "text": str(text or ""),
                "encoding": str(encoding or "utf-8"),
                "create_parents": bool(create_parents),
            },
        )

    def mkdir(self, *, root_id: str, relative_path: str, parents: bool = True, exist_ok: bool = True) -> Dict[str, Any]:
        return self.host.call(
            "fs.mkdir",
            {
                "root_id": str(root_id or ""),
                "relative_path": str(relative_path or ""),
                "parents": bool(parents),
                "exist_ok": bool(exist_ok),
            },
        )

    def stat(self, *, root_id: str, relative_path: Optional[str] = None) -> Dict[str, Any]:
        return self.host.call("fs.stat", {"root_id": str(root_id or ""), "relative_path": relative_path})


class BrokeredHttpClient:
    def __init__(self, *, host: HostCallbackClient) -> None:
        self.host = host

    def fetch(
        self,
        *,
        url: str,
        method: str = "GET",
        headers: Optional[Dict[str, str]] = None,
        body_b64: str = "",
        timeout_seconds: float = 30.0,
        max_response_bytes: int = 1024 * 1024,
    ) -> Dict[str, Any]:
        return self.host.call(
            "http.fetch",
            {
                "url": str(url or ""),
                "method": str(method or "GET"),
                "headers": dict(headers or {}),
                "body_b64": str(body_b64 or ""),
                "timeout_seconds": float(timeout_seconds or 30.0),
                "max_response_bytes": int(max_response_bytes or 1024 * 1024),
            },
        )


class GenericCallbackClient:
    def __init__(self, *, host: HostCallbackClient) -> None:
        self.host = host

    def invoke(self, callback_name: str, payload: Any = None) -> Any:
        response = self.host.call(
            "callback.invoke",
            {
                "callback_name": str(callback_name or "").strip(),
                "payload": payload,
            },
        )
        return dict(response or {}).get("result")


class ToolboxExecutionContext:
    def __init__(
        self,
        *,
        engine_id: str,
        toolbox_id: str = "",
        tool_name: str = "",
        tool_call_id: str = "",
        tool_arguments: Optional[Dict[str, Any]] = None,
        callback_binding: Optional[Dict[str, Any]] = None,
        callback_signature: Optional[Dict[str, Any]] = None,
        host_api_approval: Optional[Dict[str, Any]] = None,
    ) -> None:
        callback_binding_payload = dict(callback_binding or {})
        self.engine_id = str(engine_id or "").strip() or _worker_engine_id()
        self.host = HostCallbackClient(
            engine_id=self.engine_id,
            callback_binding=callback_binding_payload,
            toolbox_id=toolbox_id,
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            tool_arguments=tool_arguments,
            callback_signature=callback_signature,
            host_api_approval=host_api_approval,
            user_context=None,
        )
        self.fs = BrokeredFsClient(host=self.host)
        self.http = BrokeredHttpClient(host=self.host)
        self.callbacks = GenericCallbackClient(host=self.host)


def _manifest_path() -> Path:
    spec = _startup_spec_or_none()
    if spec and str(spec.manifest_path or "").strip():
        return Path(str(spec.manifest_path)).expanduser().resolve()
    raw = str(os.environ.get("MP13_TOOLBOX_MANIFEST_PATH") or "").strip()
    if not raw:
        raise RuntimeError("MP13_TOOLBOX_MANIFEST_PATH or MP13_TOOLBOX_WORKER_SPEC_PATH is required")
    return Path(raw).expanduser().resolve()


def _ensure_toolbox() -> tuple[Any, Dict[str, Any]]:
    global _toolbox_state
    with _toolbox_lock:
        if _toolbox_state is None:
            _toolbox_state = load_toolbox_from_manifest(_manifest_path())
        return _toolbox_state


def _manifest_tool_names(manifest: Dict[str, Any]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in list(manifest.get("tools") or []):
        name = str(dict(item or {}).get("name") or "").strip()
        if name and name not in seen:
            seen.add(name)
            out.append(name)
    for item in list(manifest.get("auto_tools") or []):
        name = str(dict(item or {}).get("name") or "").strip()
        if name and name not in seen:
            seen.add(name)
            out.append(name)
    for item in list(manifest.get("active_intrinsic_tool_names") or []):
        name = str(item or "").strip()
        if name and name not in seen:
            seen.add(name)
            out.append(name)
    return out


async def _handle_hello(_payload: Dict[str, Any]) -> Dict[str, Any]:
    import json
    try:
        manifest = json.loads(_manifest_path().read_text(encoding="utf-8"))
    except Exception:
        manifest = {}
    tool_names = _manifest_tool_names(manifest)
    tool_metadata = {
        str(item.get("name") or "").strip(): {
            "callback_signature": dict(item.get("callback_signature") or {}) or None,
            "non_restartable": bool(item.get("non_restartable", False)),
            "hidden": bool(item.get("hidden", False)),
        }
        for item in list(manifest.get("auto_tools") or []) + list(manifest.get("tools") or [])
        if str(dict(item or {}).get("name") or "").strip()
    }
    return {
        "status": "ok",
        "pid": os.getpid(),
        "executable": sys.executable,
        "prefix": sys.prefix,
        "protocol_version": PROTOCOL_VERSION,
        "contract": _contract_name(),
        "sync_rpc": True,
        "async_rpc": True,
        "cancellation": False,
        "all_registered_tool_names": tool_names,
        "tool_metadata": tool_metadata,
        "host_capabilities": _toolbox_host_capability_broker(engine_id=_worker_engine_id()).describe().get("host_capabilities"),
        "executor_kind": str(manifest.get("executor_kind") or "toolbox_executor"),
    }


async def _rpc_call(method: str, params: Dict[str, Any]) -> Dict[str, Any]:
    meth = str(method or "").strip()
    if meth in {"rpc.describe", "describe", "capabilities"}:
        return await _handle_hello({})
    if meth == "host.call":
        host_method = str(params.get("method") or "").strip()
        if not host_method:
            return {"status": "error", "message": "host_call_method_required"}
        arguments = params.get("arguments") if isinstance(params.get("arguments"), dict) else {}
        try:
            result = _invoke_host_call(
                host_method,
                dict(arguments or {}),
                callback_binding=dict(params.get("callback_binding") or {}) if isinstance(params.get("callback_binding"), dict) else None,
                approval=dict(params.get("approval") or {}) if isinstance(params.get("approval"), dict) else None,
            )
        except Exception as exc:
            return {"status": "error", "message": f"host_call_failed:{exc}"}
        return {"status": "ok", "result": result}
    toolbox, manifest = _ensure_toolbox()
    if meth == "toolbox.describe":
        tool_names = _manifest_tool_names(manifest)
        tool_metadata = {
            str(item.get("name") or "").strip(): {
                "callback_signature": dict(item.get("callback_signature") or {}) or None,
                "non_restartable": bool(item.get("non_restartable", False)),
                "hidden": bool(item.get("hidden", False)),
            }
            for item in list(manifest.get("auto_tools") or []) + list(manifest.get("tools") or [])
            if str(dict(item or {}).get("name") or "").strip()
        }
        return {
            "status": "ok",
            "executor_kind": str(manifest.get("executor_kind") or "toolbox_executor"),
            "bundle": {
                "bundle_id": manifest.get("bundle_id"),
                "bundle_revision": manifest.get("bundle_revision"),
                "manifest_hash": manifest.get("manifest_hash"),
            },
            "all_registered_tool_names": tool_names,
            "tool_metadata": tool_metadata,
            "host_capabilities": _toolbox_host_capability_broker(engine_id=_worker_engine_id()).describe().get("host_capabilities"),
            "parallel_execution": {
                "async_within_executor": True,
                "sandbox_pool": False,
            },
        }
    if meth == "toolbox.execute":
        raw_call = dict(params.get("tool_call") or {})
        call = ToolCall.from_dict(raw_call)
        callback_binding = dict(params.get("callback_binding") or {})
        tool_names = set(_manifest_tool_names(manifest))
        if call.name not in tool_names:
            call.error = f"Error: Tool '{call.name}' is not staged in this executor."
            return {"status": "ok", "tool_call": call.to_dict()}
        tool_def = toolbox.get_tool(call.name) or {}
        context = ToolboxExecutionContext(
            engine_id=_worker_engine_id(),
            toolbox_id=str(manifest.get("toolbox_id") or ""),
            tool_name=call.name,
            tool_call_id=str(call.id or "").strip(),
            tool_arguments=dict(call.arguments or {}),
            callback_binding=callback_binding,
            callback_signature=dict(tool_def.get("callback_signature") or {}) or None,
            host_api_approval=dict(params.get("host_api_approval") or {}) if isinstance(params.get("host_api_approval"), dict) else None,
        )
        result = await toolbox.execute(
            call,
            context=context,
            host=context.host,
            fs=context.fs,
            http=context.http,
            callbacks=context.callbacks,
        )
        if result is not None:
            call.result = result
        return {"status": "ok", "tool_call": call.to_dict()}
    return {"status": "error", "message": f"unsupported_method:{meth}"}


async def _handle_rpc_call(payload: Dict[str, Any]) -> Dict[str, Any]:
    method = str(payload.get("method") or "").strip()
    params = payload.get("params") if isinstance(payload.get("params"), dict) else {}
    if not method:
        return {"status": "error", "message": "method_required"}
    return await _rpc_call(method, dict(params or {}))


def _handle_conn(conn: Any, stop_event: threading.Event) -> None:
    try:
        req = conn.recv()
        if not isinstance(req, dict):
            conn.send({"status": "error", "message": "invalid_request"})
            return
        kind = str(req.get("kind") or "").strip().lower()
        if kind == "shutdown":
            conn.send({"status": "ok"})
            stop_event.set()
            return
        if kind == "hello":
            conn.send(asyncio.run(_handle_hello(req)))
            return
        if kind == "rpc_call":
            conn.send(asyncio.run(_handle_rpc_call(req)))
            return
        conn.send({"status": "error", "message": "unsupported_kind"})
    except Exception as exc:
        try:
            conn.send({"status": "error", "message": f"worker_exception:{exc}"})
        except Exception:
            pass
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _serve_loop(*, family: str, address: str, authkey: bytes) -> int:
    listener = None
    unix_path = Path(address) if family == "AF_UNIX" else None
    stop_event = threading.Event()
    workers: list[threading.Thread] = []
    accepted: "queue.Queue[Any]" = queue.Queue()
    accept_errors: "queue.Queue[BaseException]" = queue.Queue()

    def _accept_loop() -> None:
        assert listener is not None
        while not stop_event.is_set():
            try:
                conn = listener.accept()
            except socket.timeout:
                continue
            except OSError as exc:
                if stop_event.is_set():
                    break
                accept_errors.put(exc)
                break
            except Exception as exc:
                if stop_event.is_set():
                    break
                accept_errors.put(exc)
                break
            accepted.put(conn)

    if unix_path is not None:
        try:
            if unix_path.exists():
                unix_path.unlink()
        except Exception:
            pass
    accept_thread: Optional[threading.Thread] = None
    try:
        _ensure_toolbox()
        listener = Listener(address=address, family=family, authkey=authkey)
        try:
            raw_sock = getattr(getattr(listener, "_listener", None), "_socket", None)
            if raw_sock is not None:
                raw_sock.settimeout(0.5)
        except Exception:
            pass
        accept_thread = threading.Thread(target=_accept_loop, daemon=True)
        accept_thread.start()
        while not stop_event.is_set():
            try:
                if not accept_errors.empty():
                    raise accept_errors.get()
                conn = accepted.get(timeout=0.2)
            except queue.Empty:
                continue
            t = threading.Thread(target=_handle_conn, args=(conn, stop_event), daemon=True)
            t.start()
            workers.append(t)
    finally:
        if listener is not None:
            try:
                listener.close()
            except Exception:
                pass
        if accept_thread is not None:
            accept_thread.join(timeout=1.0)
        for t in workers[-256:]:
            t.join(timeout=0.5)
        if unix_path is not None:
            try:
                if unix_path.exists():
                    unix_path.unlink()
            except Exception:
                pass
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ipc-family", required=True, choices=["AF_UNIX", "AF_PIPE"])
    ap.add_argument("--ipc-address", required=True)
    args = ap.parse_args()

    auth_token = str(os.environ.get("MP13_ENGINE_HOST_TOKEN") or "").strip()
    if not auth_token:
        print("Missing MP13_ENGINE_HOST_TOKEN", flush=True)
        return 2
    try:
        _ensure_toolbox()
    except Exception:
        print(traceback.format_exc(), flush=True)
        return 3
    return _serve_loop(
        family=str(args.ipc_family),
        address=str(args.ipc_address),
        authkey=auth_token.encode("utf-8", errors="ignore"),
    )


if __name__ == "__main__":
    raise SystemExit(main())
