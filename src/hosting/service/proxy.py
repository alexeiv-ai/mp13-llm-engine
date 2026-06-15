"""IPC and proxy helpers for the engine host service."""
from __future__ import annotations

import base64
import hashlib
import os
import posixpath
import re
import secrets
import sys
import tempfile
import time
from multiprocessing.connection import Client as MPClient
from typing import Any, Dict, Optional, Tuple


class ProxyMixin:
    @staticmethod
    def _allocate_ipc_address(engine_id: str) -> Tuple[str, str]:
        raw_engine = str(engine_id or "engine")
        safe_engine = re.sub(r"[^A-Za-z0-9_-]+", "_", raw_engine).strip("_") or "engine"
        nonce = secrets.token_hex(6)
        if os.name == "nt":
            return "AF_PIPE", f"\\\\.\\pipe\\mp13-host-{safe_engine}-{nonce}"
        base = posixpath.abspath(posixpath.expanduser(str(tempfile.gettempdir() or "/tmp")))
        engine_hash = hashlib.sha256(raw_engine.encode("utf-8", errors="ignore")).hexdigest()[:12]
        short_engine = safe_engine[:24].rstrip("_-") or "engine"
        filename = f"mp13-host-{short_engine}-{engine_hash}-{nonce}.sock"
        return "AF_UNIX", posixpath.join(base, filename)

    @staticmethod
    def _parse_worker_authkey_token(token: Optional[str]) -> bytes:
        raw = str(token or "").strip()
        if not raw:
            return b""
        return raw.encode("utf-8", errors="ignore")

    def _proxy_request_via_ipc(
        self,
        *,
        reg: Dict[str, Any],
        engine_id: str,
        method: str,
        path: str,
        query: str,
        headers: Dict[str, str],
        body_b64: str,
        timeout_seconds: float,
    ) -> Dict[str, Any]:
        family = str(reg.get("worker_ipc_family") or "").strip()
        address = str(reg.get("worker_ipc_address") or "").strip()
        auth_token = str(reg.get("worker_auth_token") or "").strip()
        import socket
        endpoint = str(reg.get("endpoint") or "").strip() or f"ipc://{socket.gethostname()}"
        if not family or not address:
            raise ValueError("engine ipc endpoint is not registered")
        authkey = self._parse_worker_authkey_token(auth_token)
        payload = {
            "kind": "http_request",
            "engine_id": str(engine_id or "").strip(),
            "method": str(method or "GET").strip().upper(),
            "path": str(path or "/").strip() or "/",
            "query": str(query or ""),
            "headers": dict(headers or {}),
            "body_b64": str(body_b64 or ""),
        }
        conn = None
        try:
            conn = MPClient(address=address, family=family, authkey=authkey)
            conn.send(payload)
            if not conn.poll(max(0.1, float(timeout_seconds or 30.0))):
                raise TimeoutError("ipc worker timeout")
            resp = conn.recv()
        finally:
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass
        if not isinstance(resp, dict):
            raise RuntimeError("invalid ipc worker response")
        if str(resp.get("status") or "").strip().lower() == "error":
            msg = str(resp.get("message") or "ipc worker error")
            raise RuntimeError(msg)
        status_code = int(resp.get("status_code") or 500)
        out_headers = dict(resp.get("headers") or {})
        out_body_b64 = str(resp.get("body_b64") or "")
        return {
            "engine_id": str(engine_id),
            "endpoint": endpoint,
            "url": f"ipc://{engine_id}{path}",
            "status_code": status_code,
            "headers": out_headers,
            "body_b64": out_body_b64,
            "body_size": len(base64.b64decode(out_body_b64)) if out_body_b64 else 0,
            "truncated": False,
        }

    def _ipc_call(self, *, reg: Dict[str, Any], payload: Dict[str, Any], timeout_seconds: float = 30.0) -> Dict[str, Any]:
        family = str(reg.get("worker_ipc_family") or "").strip()
        address = str(reg.get("worker_ipc_address") or "").strip()
        auth_token = str(reg.get("worker_auth_token") or "").strip()
        if not family or not address:
            raise ValueError("engine ipc endpoint is not registered")
        authkey = self._parse_worker_authkey_token(auth_token)
        conn = None
        try:
            conn = MPClient(address=address, family=family, authkey=authkey)
            conn.send(dict(payload or {}))
            if not conn.poll(max(0.1, float(timeout_seconds or 30.0))):
                raise TimeoutError("ipc worker timeout")
            out = conn.recv()
            if not isinstance(out, dict):
                raise RuntimeError("invalid ipc response")
            return dict(out or {})
        except FileNotFoundError as exc:
            engine_id = str(reg.get("engine_id") or "").strip() or "unknown"
            raise RuntimeError(
                f"worker IPC endpoint is unavailable for engine '{engine_id}' at '{address}'; "
                "worker process may not be running"
            ) from exc
        finally:
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass

    def _require_ipc_registration(self, engine_id: str, *, command_label: str) -> Dict[str, Any]:
        eid = str(engine_id or "").strip()
        reg = self._find_registration(eid)
        if not reg:
            raise ValueError(f"engine '{eid}' is not registered")
        if str(reg.get("worker_transport") or "").strip().lower() != "ipc":
            raise ValueError(f"{command_label} is only supported for ipc transport")
        return reg

    def _route_model_instance_id(self, reg: Dict[str, Any], engine_id: str) -> str:
        routed = str(reg.get("_route_model_instance_id") or "").strip()
        if routed:
            return routed
        try:
            return self._model_instance_for_engine_id(reg, engine_id)
        except Exception:
            return str(engine_id or "").strip()

    def proxy_request(
        self,
        *,
        engine_id: str,
        method: str = "GET",
        path: str = "/",
        query: str = "",
        headers: Optional[Dict[str, str]] = None,
        body_b64: str = "",
        timeout_seconds: float = 30.0,
        max_response_bytes: int = 1024 * 1024,
    ) -> Dict[str, Any]:
        eid = str(engine_id or "").strip()
        req_started_at = time.time()
        m = str(method or "GET").strip().upper()
        req_path = str(path or "/").strip() or "/"
        if not req_path.startswith("/"):
            req_path = f"/{req_path}"
        if not eid:
            raise ValueError("engine_id is required")
        reg = self._find_registration(eid) or {}
        worker_engine_id = self._route_model_instance_id(reg, eid) if reg else eid
        endpoint = str(reg.get("endpoint") or "").strip()
        if not endpoint:
            self._metrics_proxy_finish(
                eid,
                failed=True,
                error_message="engine endpoint is not registered",
                method=m,
                path=req_path,
                started_at=req_started_at,
            )
            raise ValueError("engine endpoint is not registered")
        transport = str(reg.get("worker_transport") or "").strip().lower()
        if transport != "ipc":
            self._metrics_proxy_finish(
                eid,
                failed=True,
                error_message="ipc transport is required",
                method=m,
                path=req_path,
                started_at=req_started_at,
            )
            raise ValueError("ipc transport is required")
        traffic_policy = self._traffic_policy_for_engine(eid)
        if not re.fullmatch(r"[A-Z]+", m):
            self._metrics_proxy_finish(
                eid,
                failed=True,
                error_message="invalid method",
                method=m,
                path=req_path,
                started_at=req_started_at,
            )
            raise ValueError("invalid method")
        allowed_methods = set(str(x).upper() for x in list(traffic_policy.get("allowed_methods") or []))
        if allowed_methods and m not in allowed_methods:
            self._metrics_proxy_finish(
                eid,
                failed=True,
                error_message=f"proxy_method_not_allowed:{m}",
                method=m,
                path=req_path,
                started_at=req_started_at,
            )
            raise PermissionError(f"proxy_method_not_allowed:{m}")
        prefixes = [str(x) for x in list(traffic_policy.get("allowed_path_prefixes") or ["/"])]
        if prefixes and not any(req_path.startswith(px if px else "/") for px in prefixes):
            self._metrics_proxy_finish(
                eid,
                failed=True,
                error_message=f"proxy_path_not_allowed:{req_path}",
                method=m,
                path=req_path,
                started_at=req_started_at,
            )
            raise PermissionError(f"proxy_path_not_allowed:{req_path}")
        body_raw = b""
        if str(body_b64 or "").strip():
            try:
                body_raw = base64.b64decode(str(body_b64), validate=True)
            except Exception as exc:
                self._metrics_proxy_finish(
                    eid,
                    failed=True,
                    error_message=f"invalid body_b64: {exc}",
                    method=m,
                    path=req_path,
                    started_at=req_started_at,
                )
                raise ValueError(f"invalid body_b64: {exc}") from exc
        max_req = int(traffic_policy.get("max_request_bytes") or (1024 * 1024))
        if len(body_raw) > max_req:
            self._metrics_proxy_finish(
                eid,
                failed=True,
                error_message=f"request body too large ({len(body_raw)} > {max_req})",
                method=m,
                path=req_path,
                started_at=req_started_at,
                request_bytes=len(body_raw),
            )
            raise ValueError(f"request body too large ({len(body_raw)} > {max_req})")
        self._metrics_proxy_start(eid, request_bytes=len(body_raw))
        header_allow = set(str(x).lower() for x in list(traffic_policy.get("request_header_allowlist") or []))
        allow_authz = bool(traffic_policy.get("allow_authorization_header", False))
        req_headers: Dict[str, str] = {}
        for k, v in dict(headers or {}).items():
            key = str(k or "").strip()
            if not key:
                continue
            low = key.lower()
            if low == "authorization" and not allow_authz:
                continue
            if header_allow and low not in header_allow:
                continue
            req_headers[key] = str(v)
        worker_auth_header = str(reg.get("worker_auth_header") or "").strip()
        worker_auth_token = str(reg.get("worker_auth_token") or "").strip()
        if worker_auth_header and worker_auth_token:
            # Host-controlled channel proof. Client headers cannot override this.
            req_headers[worker_auth_header] = worker_auth_token
        try:
            out = self._proxy_request_via_ipc(
                reg=reg,
                engine_id=worker_engine_id,
                method=m,
                path=req_path,
                query=query,
                headers=req_headers,
                body_b64=str(body_b64 or ""),
                timeout_seconds=timeout_seconds,
            )
            out["engine_id"] = eid
            out["worker_engine_id"] = worker_engine_id
            raw = base64.b64decode(str(out.get("body_b64") or "")) if str(out.get("body_b64") or "") else b""
            lim = min(
                max(1024, int(max_response_bytes or 1024 * 1024)),
                max(1024, int(traffic_policy.get("max_response_bytes") or (1024 * 1024))),
            )
            truncated = len(raw) > lim
            if truncated:
                raw = raw[:lim]
                out["body_b64"] = base64.b64encode(raw).decode("ascii")
                out["body_size"] = len(raw)
                out["truncated"] = True
            self._metrics_proxy_finish(
                eid,
                status_code=int(out.get("status_code") or 500),
                response_bytes=len(raw),
                http_error=bool(int(out.get("status_code") or 500) >= 400),
                failed=False,
                method=m,
                path=req_path,
                started_at=req_started_at,
                truncated=bool(out.get("truncated")),
                request_bytes=len(body_raw),
            )
            return out
        except Exception as exc:
            self._metrics_proxy_finish(
                eid,
                failed=True,
                error_message=str(exc),
                method=m,
                path=req_path,
                started_at=req_started_at,
                request_bytes=len(body_raw),
            )
            raise
        finally:
            # Ensure we decrement inflight in paths where finish wasn't called yet.
            with self._metrics_lock:
                assert isinstance(self._runtime_metrics, dict)
                proxy = dict(self._runtime_metrics.get("proxy") or {})
                inflight_by_engine = dict(proxy.get("inflight_by_engine") or {})
                current = int(inflight_by_engine.get(eid) or 0)
                if current > 0:
                    if current == 1:
                        inflight_by_engine.pop(eid, None)
                    else:
                        inflight_by_engine[eid] = current - 1
                    proxy["inflight_by_engine"] = inflight_by_engine
                    proxy["inflight_total"] = max(0, int(proxy.get("inflight_total") or 0) - 1)
                    self._runtime_metrics["proxy"] = proxy

    def proxy_rpc_call(
        self,
        *,
        engine_id: str,
        method: str,
        params: Optional[Dict[str, Any]] = None,
        timeout_seconds: float = 30.0,
    ) -> Dict[str, Any]:
        eid = str(engine_id or "").strip()
        if not eid:
            raise ValueError("engine_id is required")
        meth = str(method or "").strip()
        if not meth:
            raise ValueError("method is required")
        reg = self._require_ipc_registration(eid, command_label="proxy-rpc")
        worker_engine_id = self._route_model_instance_id(reg, eid)
        rpc_params = dict(params or {})
        workflow_python_facade_execute = bool(rpc_params.pop("_workflow_python_facade_execute", False))
        if (
            str(reg.get("executor_kind") or "").strip() == "workflow_python_helper"
            and meth == "execute_workflow_python_helper"
            and not workflow_python_facade_execute
        ):
            capacity = int(dict(reg.get("capabilities") or {}).get("capacity") or 1)
            facade = self.execute_workflow_python(
                profile="helper",
                engine_id=eid,
                request=rpc_params,
                capacity=capacity,
                sandbox_policy=dict(reg.get("sandbox_policy") or {}) or None,
            )
            return {
                "status": str(facade.get("status") or "ok"),
                "result": dict(facade.get("result") or {}),
                "workflow_runtime_kind": "workflow_python",
                "workflow_profile": "helper",
                "environment_key": facade.get("environment_key"),
                "workflow_execute": facade,
            }
        if str(reg.get("executor_kind") or "").strip() == "workflow_python_helper" and meth == "execute_workflow_python_helper":
            rpc_params = self._prepare_workflow_python_helper_runtime_params(reg=reg, params=rpc_params)
        out = self._ipc_call(
            reg=reg,
            payload={"kind": "rpc_call", "engine_id": worker_engine_id, "method": meth, "params": rpc_params},
            timeout_seconds=timeout_seconds,
        )
        if str(out.get("status") or "").strip().lower() == "error":
            raise RuntimeError(str(out.get("message") or "rpc_call_failed"))
        return dict(out or {})

    def _prepare_workflow_python_helper_runtime_params(self, *, reg: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(params or {})
        python_req = dict(out.get("python") or {})
        package_pins = dict(python_req.get("package_pins") or {})
        import_allowlist = list(python_req.get("import_allowlist") or [])
        environment_name = str(python_req.get("environment_name") or "workflow-python-helper").strip() or "workflow-python-helper"
        if not package_pins and not import_allowlist:
            python_req.setdefault("environment_name", environment_name)
            python_req.setdefault("python_executable", str(dict(reg.get("env") or {}).get("MP13_WORKFLOW_PYTHON") or sys.executable))
            python_req.setdefault("python_source", "worker")
            out["python"] = python_req
            return out
        from ..toolbox.environment import RuntimeEnvironmentManager

        manager = RuntimeEnvironmentManager(self.hosting_root)
        metadata = manager.realize_workflow_python_helper_environment(
            policy={
                "import_allowlist": import_allowlist,
                "package_pins": package_pins,
            },
            package_id=str(out.get("package_id") or "").strip() or None,
            workflow_id=str(out.get("workflow_id") or "").strip() or None,
            package_source_digest=str(out.get("package_source_digest") or "").strip() or None,
            helper_source_sha256=str(out.get("module_sha256") or "").strip() or None,
            helper_source_path=str(out.get("source_path") or "").strip() or None,
            bootstrap_python_executable=str(dict(reg.get("env") or {}).get("MP13_WORKFLOW_PYTHON") or sys.executable),
            environment_name=environment_name,
        )
        python_req["environment_name"] = environment_name
        python_req["python_executable"] = str(metadata.get("runtime_python_executable") or python_req.get("python_executable") or sys.executable)
        python_req["python_source"] = str(metadata.get("runtime_python_source") or "runtime_env")
        python_req["runtime_environment"] = {
            "venv_key": str(metadata.get("venv_key") or "").strip() or None,
            "venv_path": str(metadata.get("venv_path") or "").strip() or None,
            "runtime_python_selection": dict(metadata.get("runtime_python_selection") or {}),
        }
        out["python"] = python_req
        return out

    def proxy_rpc_open(
        self,
        *,
        engine_id: str,
        method: str,
        params: Optional[Dict[str, Any]] = None,
        request_id: str,
        timeout_seconds: float = 30.0,
    ) -> Dict[str, Any]:
        eid = str(engine_id or "").strip()
        if not eid:
            raise ValueError("engine_id is required")
        meth = str(method or "").strip()
        if not meth:
            raise ValueError("method is required")
        req_id = str(request_id or "").strip()
        if not req_id:
            raise ValueError("request_id is required")
        reg = self._require_ipc_registration(eid, command_label="proxy-rpc")
        worker_engine_id = self._route_model_instance_id(reg, eid)
        out = self._ipc_call(
            reg=reg,
            payload={
                "kind": "stream_open",
                "engine_id": worker_engine_id,
                "method": meth,
                "params": dict(params or {}),
                "request_id": req_id,
            },
            timeout_seconds=timeout_seconds,
        )
        if str(out.get("status") or "").strip().lower() == "error":
            raise RuntimeError(str(out.get("message") or "rpc_open_failed"))
        return {"status": "ok", "engine_id": eid, "worker_engine_id": worker_engine_id, "stream_id": str(out.get("stream_id") or ""), "request_id": req_id}

    def proxy_rpc_send(
        self,
        *,
        engine_id: str,
        stream_id: str,
        message: Optional[Dict[str, Any]] = None,
        timeout_seconds: float = 30.0,
    ) -> Dict[str, Any]:
        eid = str(engine_id or "").strip()
        sid = str(stream_id or "").strip()
        if not eid:
            raise ValueError("engine_id is required")
        if not sid:
            raise ValueError("stream_id is required")
        reg = self._require_ipc_registration(eid, command_label="proxy-rpc")
        worker_engine_id = self._route_model_instance_id(reg, eid)
        out = self._ipc_call(
            reg=reg,
            payload={"kind": "stream_send", "engine_id": worker_engine_id, "stream_id": sid, "message": dict(message or {})},
            timeout_seconds=timeout_seconds,
        )
        if str(out.get("status") or "").strip().lower() == "error":
            raise RuntimeError(str(out.get("message") or "rpc_send_failed"))
        return dict(out or {})

    def proxy_rpc_recv(
        self,
        *,
        engine_id: str,
        stream_id: str,
        timeout_seconds: float = 2.0,
        max_items: int = 64,
    ) -> Dict[str, Any]:
        eid = str(engine_id or "").strip()
        sid = str(stream_id or "").strip()
        if not eid:
            raise ValueError("engine_id is required")
        if not sid:
            raise ValueError("stream_id is required")
        reg = self._require_ipc_registration(eid, command_label="proxy-rpc")
        worker_engine_id = self._route_model_instance_id(reg, eid)
        out = self._ipc_call(
            reg=reg,
            payload={
                "kind": "stream_recv",
                "engine_id": worker_engine_id,
                "stream_id": sid,
                "timeout_seconds": float(timeout_seconds or 2.0),
                "max_items": int(max_items or 64),
            },
            timeout_seconds=max(1.0, float(timeout_seconds or 2.0) + 1.0),
        )
        if str(out.get("status") or "").strip().lower() == "error":
            raise RuntimeError(str(out.get("message") or "rpc_recv_failed"))
        return dict(out or {})

    def proxy_rpc_close(
        self,
        *,
        engine_id: str,
        stream_id: str,
        timeout_seconds: float = 10.0,
    ) -> Dict[str, Any]:
        eid = str(engine_id or "").strip()
        sid = str(stream_id or "").strip()
        if not eid:
            raise ValueError("engine_id is required")
        if not sid:
            raise ValueError("stream_id is required")
        reg = self._require_ipc_registration(eid, command_label="proxy-rpc")
        worker_engine_id = self._route_model_instance_id(reg, eid)
        out = self._ipc_call(
            reg=reg,
            payload={"kind": "stream_close", "engine_id": worker_engine_id, "stream_id": sid},
            timeout_seconds=timeout_seconds,
        )
        if str(out.get("status") or "").strip().lower() == "error":
            raise RuntimeError(str(out.get("message") or "rpc_close_failed"))
        return dict(out or {})

    def proxy_stream_open(
        self,
        *,
        engine_id: str,
        tool: str = "run-inference",
        arguments: Optional[Dict[str, Any]] = None,
        timeout_seconds: float = 30.0,
    ) -> Dict[str, Any]:
        args = dict(arguments or {})
        req_id = str(args.get("request_id") or "").strip() or secrets.token_hex(12)
        out = self.proxy_rpc_open(
            engine_id=str(engine_id or ""),
            method=str(tool or "run-inference"),
            params=args,
            request_id=req_id,
            timeout_seconds=timeout_seconds,
        )
        out["worker_transport"] = "ipc"
        return out

    def proxy_stream_send(
        self,
        *,
        engine_id: str,
        stream_id: str,
        message: Optional[Dict[str, Any]] = None,
        timeout_seconds: float = 30.0,
    ) -> Dict[str, Any]:
        return self.proxy_rpc_send(
            engine_id=str(engine_id or ""),
            stream_id=str(stream_id or ""),
            message=dict(message or {}),
            timeout_seconds=timeout_seconds,
        )

    def proxy_stream_recv(
        self,
        *,
        engine_id: str,
        stream_id: str,
        timeout_seconds: float = 2.0,
        max_items: int = 64,
    ) -> Dict[str, Any]:
        return self.proxy_rpc_recv(
            engine_id=str(engine_id or ""),
            stream_id=str(stream_id or ""),
            timeout_seconds=float(timeout_seconds or 2.0),
            max_items=int(max_items or 64),
        )

    def proxy_stream_close(
        self,
        *,
        engine_id: str,
        stream_id: str,
        timeout_seconds: float = 10.0,
    ) -> Dict[str, Any]:
        return self.proxy_rpc_close(
            engine_id=str(engine_id or ""),
            stream_id=str(stream_id or ""),
            timeout_seconds=timeout_seconds,
        )

    def _revoke_engine_tokens(self, control: Dict[str, Any], engine_id: str) -> int:
        tokens = dict(control.get("tokens") or {})
        revoked = 0
        for token, meta in list(tokens.items()):
            if str((meta or {}).get("engine_id") or "") == str(engine_id):
                tokens.pop(token, None)
                revoked += 1
        control["tokens"] = tokens
        return revoked

    def _revoke_all_tokens(self, control: Dict[str, Any]) -> int:
        t = dict(control.get("tokens") or {})
        r = dict(control.get("resource_tokens") or {})
        revoked = len(t) + len(r)
        control["tokens"] = {}
        control["resource_tokens"] = {}
        return revoked
