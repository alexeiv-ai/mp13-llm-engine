from __future__ import annotations

import asyncio
import base64
import http.client
import json
import os
import re
import socket
import threading
import time
from pathlib import Path

from hosting.daemon import EngineHostDaemon, EngineHostHttpIngressDaemon
from hosting.service.host_service import EngineHostService


def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = int(s.getsockname()[1])
    s.close()
    return port


def _wait_health(port: int, timeout: float = 6.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=1.0)
            conn.request("GET", "/health")
            resp = conn.getresponse()
            _ = resp.read()
            conn.close()
            if int(resp.status) == 200:
                return
        except Exception:
            pass
        time.sleep(0.1)
    raise RuntimeError(f"http ingress not ready on {port}")


def _install_ipc_http_stub(monkeypatch) -> None:
    def _stub(
        self,
        *,
        reg,
        engine_id: str,
        method: str,
        path: str,
        query: str,
        headers: dict[str, str],
        body_b64: str,
        timeout_seconds: float,
    ) -> dict[str, object]:
        body = b'{"ok":true}' if str(path).startswith("/health") else b"not found"
        status = 200 if str(path).startswith("/health") else 404
        return {
            "engine_id": str(engine_id),
            "endpoint": "ipc://local",
            "url": f"ipc://{engine_id}{path}",
            "status_code": status,
            "headers": {"content-type": "application/json" if status == 200 else "text/plain"},
            "body_b64": base64.b64encode(body).decode("ascii"),
            "body_size": len(body),
            "truncated": False,
        }

    monkeypatch.setattr(EngineHostService, "_proxy_request_via_ipc", _stub)


def test_http_ingress_proxy_with_traffic_auth(tmp_path: Path, monkeypatch) -> None:
    engines_state = tmp_path / "managed_engines.json"
    control_state = tmp_path / "access_control.json"
    pid_file = tmp_path / "daemon_http.pid"

    _install_ipc_http_stub(monkeypatch)
    svc = EngineHostService(engines_state_file=engines_state, control_state_file=control_state)
    svc.register_spawned(
        engine_id="worker1",
        pid=os.getpid(),
        command=["python", "-m", "hosting.engine_worker_ipc"],
    )
    svc.auth_upsert_key(
        key_id="traffic1",
        key_secret="secret1",
        role="model_user",
        allowed_engines=["worker1"],
    )
    svc.set_control_config(require_auth=True)
    issued = svc.auth_issue_session(
        key_id="traffic1",
        key_secret="secret1",
        scope="traffic",
        engine_ids=["worker1"],
        ttl_seconds=300,
    )
    token = str(issued["token"])

    ingress_port = _free_port()
    daemon = EngineHostHttpIngressDaemon(
        port=ingress_port,
        pid_file=pid_file,
        engines_state_file=engines_state,
        control_state_file=control_state,
    )
    t = threading.Thread(target=daemon.run, daemon=True)
    t.start()
    _wait_health(ingress_port)
    try:
        conn = http.client.HTTPConnection("127.0.0.1", ingress_port, timeout=4.0)
        conn.request(
            "GET",
            "/proxy/worker1/health",
            headers={"Authorization": f"Bearer {token}"},
        )
        resp = conn.getresponse()
        raw = resp.read()
        conn.close()
        assert int(resp.status) == 200
        payload = json.loads(raw.decode("utf-8"))
        assert payload["ok"] is True
        result = dict(payload["result"])
        assert int(result["status_code"]) == 200
        body = base64.b64decode(str(result["body_b64"]))
        assert b'"ok":true' in body

        # No token -> unauthorized
        conn = http.client.HTTPConnection("127.0.0.1", ingress_port, timeout=4.0)
        conn.request("GET", "/proxy/worker1/health")
        resp = conn.getresponse()
        _ = resp.read()
        conn.close()
        assert int(resp.status) == 401

        # Valid token but engine not allowed -> forbidden
        conn = http.client.HTTPConnection("127.0.0.1", ingress_port, timeout=4.0)
        conn.request(
            "GET",
            "/proxy/worker2/health",
            headers={"Authorization": f"Bearer {token}"},
        )
        resp = conn.getresponse()
        _ = resp.read()
        conn.close()
        assert int(resp.status) == 403
    finally:
        conn = http.client.HTTPConnection("127.0.0.1", ingress_port, timeout=4.0)
        shutdown_payload = json.dumps({"shutdown_token": daemon.shutdown_token}).encode("utf-8")
        conn.request(
            "POST",
            "/__shutdown__",
            body=shutdown_payload,
            headers={"Content-Type": "application/json"},
        )
        resp = conn.getresponse()
        _ = resp.read()
        conn.close()
        t.join(timeout=5.0)
        assert not t.is_alive()


def test_http_ingress_per_engine_traffic_policy_override(tmp_path: Path, monkeypatch) -> None:
    engines_state = tmp_path / "managed_engines.json"
    control_state = tmp_path / "access_control.json"
    pid_file = tmp_path / "daemon_http.pid"

    _install_ipc_http_stub(monkeypatch)
    svc = EngineHostService(engines_state_file=engines_state, control_state_file=control_state)
    svc.register_spawned(
        engine_id="worker1",
        pid=os.getpid(),
        command=["python", "-m", "hosting.engine_worker_ipc"],
    )
    svc.register_spawned(
        engine_id="worker2",
        pid=os.getpid(),
        command=["python", "-m", "hosting.engine_worker_ipc"],
    )
    svc.auth_upsert_key(
        key_id="traffic1",
        key_secret="secret1",
        role="model_user",
        allowed_engines=["worker1", "worker2"],
    )
    svc.set_control_config(
        require_auth=True,
        traffic_policy={
            "allowed_methods": ["GET"],
            "allowed_path_prefixes": ["/health"],
        },
        engine_traffic_policies={
            "worker2": {
                "allowed_methods": ["GET"],
                "allowed_path_prefixes": ["/other"],
            }
        },
    )
    issued = svc.auth_issue_session(
        key_id="traffic1",
        key_secret="secret1",
        scope="traffic",
        engine_ids=["worker1", "worker2"],
        ttl_seconds=300,
    )
    token = str(issued["token"])

    ingress_port = _free_port()
    daemon = EngineHostHttpIngressDaemon(
        port=ingress_port,
        pid_file=pid_file,
        engines_state_file=engines_state,
        control_state_file=control_state,
    )
    t = threading.Thread(target=daemon.run, daemon=True)
    t.start()
    _wait_health(ingress_port)
    try:
        conn = http.client.HTTPConnection("127.0.0.1", ingress_port, timeout=4.0)
        conn.request(
            "GET",
            "/proxy/worker1/other",
            headers={"Authorization": f"Bearer {token}"},
        )
        resp = conn.getresponse()
        _ = resp.read()
        conn.close()
        assert int(resp.status) == 403

        conn = http.client.HTTPConnection("127.0.0.1", ingress_port, timeout=4.0)
        conn.request(
            "GET",
            "/proxy/worker2/other",
            headers={"Authorization": f"Bearer {token}"},
        )
        resp = conn.getresponse()
        raw = resp.read()
        conn.close()
        assert int(resp.status) == 200
        payload = json.loads(raw.decode("utf-8"))
        assert payload["ok"] is True
        assert int(dict(payload["result"]).get("status_code") or 0) == 404
    finally:
        conn = http.client.HTTPConnection("127.0.0.1", ingress_port, timeout=4.0)
        shutdown_payload = json.dumps({"shutdown_token": daemon.shutdown_token}).encode("utf-8")
        conn.request(
            "POST",
            "/__shutdown__",
            body=shutdown_payload,
            headers={"Content-Type": "application/json"},
        )
        resp = conn.getresponse()
        _ = resp.read()
        conn.close()
        t.join(timeout=5.0)
        assert not t.is_alive()


def test_daemon_version_contract_semver_and_path_consistency(tmp_path: Path) -> None:
    engines_state = tmp_path / "managed_engines.json"
    control_state = tmp_path / "access_control.json"
    pid_file = tmp_path / "daemon_http.pid"

    daemon_rpc = EngineHostDaemon(
        port=0,
        pid_file=tmp_path / "daemon.pid",
        engines_state_file=engines_state,
        control_state_file=control_state,
    )
    req = json.dumps({"seq": 1, "cmd": "auth-status", "payload": {}})
    rpc_out = asyncio.run(daemon_rpc._dispatch(req, peer_host="127.0.0.1"))
    assert rpc_out["ok"] is True
    rpc_result = dict(rpc_out.get("result") or {})
    daemon_version = str(rpc_result.get("daemon_version") or "")
    assert re.fullmatch(r"\d+\.\d+\.\d+", daemon_version)
    assert isinstance(rpc_result.get("capabilities"), dict)

    ingress_port = _free_port()
    daemon_http = EngineHostHttpIngressDaemon(
        port=ingress_port,
        pid_file=pid_file,
        engines_state_file=engines_state,
        control_state_file=control_state,
    )
    t = threading.Thread(target=daemon_http.run, daemon=True)
    t.start()
    _wait_health(ingress_port)
    try:
        conn = http.client.HTTPConnection("127.0.0.1", ingress_port, timeout=4.0)
        conn.request("GET", "/health")
        resp = conn.getresponse()
        raw = resp.read()
        conn.close()
        assert int(resp.status) == 200
        health = json.loads(raw.decode("utf-8"))
        assert str(health.get("daemon_version") or "") == daemon_version
        assert dict(health.get("capabilities") or {}) == dict(rpc_result.get("capabilities") or {})
    finally:
        conn = http.client.HTTPConnection("127.0.0.1", ingress_port, timeout=4.0)
        shutdown_payload = json.dumps({"shutdown_token": daemon_http.shutdown_token}).encode("utf-8")
        conn.request(
            "POST",
            "/__shutdown__",
            body=shutdown_payload,
            headers={"Content-Type": "application/json"},
        )
        resp = conn.getresponse()
        _ = resp.read()
        conn.close()
        t.join(timeout=5.0)
        assert not t.is_alive()
