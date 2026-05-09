from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

from hosting.daemon import EngineHostDaemon


def _make_daemon(tmp_path: Path) -> EngineHostDaemon:
    return EngineHostDaemon(
        port=0,
        pid_file=tmp_path / "daemon.pid",
        engines_state_file=tmp_path / "managed_engines.json",
        control_state_file=tmp_path / "access_control.json",
    )


def _dispatch(daemon: EngineHostDaemon, *, seq: int, cmd: str, payload: dict, peer_host: str = "127.0.0.1") -> dict:
    raw = json.dumps({"seq": int(seq), "cmd": str(cmd), "payload": dict(payload)})
    return asyncio.run(daemon._dispatch(raw, peer_host=peer_host))


async def _dispatch_async(daemon: EngineHostDaemon, *, seq: int, cmd: str, payload: dict, peer_host: str = "127.0.0.1") -> dict:
    raw = json.dumps({"seq": int(seq), "cmd": str(cmd), "payload": dict(payload)})
    return await daemon._dispatch(raw, peer_host=peer_host)


def _issue_mgmt_session(daemon: EngineHostDaemon, key_id: str, key_secret: str) -> str:
    daemon.svc.auth_upsert_key(key_id=key_id, key_secret=key_secret, role="admin")
    issued = daemon.svc.auth_issue_session(
        key_id=key_id,
        key_secret=key_secret,
        scope="control",
        ttl_seconds=600,
    )
    return str(issued["token"])


def test_daemon_unauthorized_command_denied(tmp_path: Path) -> None:
    daemon = _make_daemon(tmp_path)
    daemon.svc.auth_upsert_key(key_id="admin", key_secret="secret", role="admin")
    daemon.svc.set_control_config(require_auth=True)

    out = _dispatch(daemon, seq=1, cmd="discover-running", payload={})
    assert out["ok"] is False
    assert out["error"] == "auth_failed"
    assert out["error_code"] == "session_token_required"


def test_daemon_lists_live_consumers(tmp_path: Path) -> None:
    daemon = _make_daemon(tmp_path)
    daemon.svc.set_control_config(require_auth=True)
    token = _issue_mgmt_session(daemon, "admin-main", "secret")
    actor_id = daemon.svc.resolve_actor_id_from_session_token(token)
    assert actor_id
    daemon._register_live_connection(
        "conn-1",
        transport="local_ipc",
        peer_host="127.0.0.1",
        pid=1234,
        process_info={"pid": 1234, "consumer_kind": "consumer"},
    )
    daemon._track_actor_connected(actor_id)
    daemon._update_live_connection(
        "conn-1",
        command="discover-running",
        actor_id=actor_id,
        session_token=token,
    )

    out = _dispatch(
        daemon,
        seq=1,
        cmd="list-live-consumers",
        payload={"session_token": token},
    )

    assert out["ok"] is True
    result = dict(out["result"])
    assert result["connections_count"] == 1
    assert result["actors_count"] == 1
    assert result["connections"][0]["connection_id"] == "conn-1"
    assert result["connections"][0]["pid"] == 1234
    assert result["connections"][0]["consumer_kind"] == "consumer"
    assert result["connections"][0]["actor_ids"] == [actor_id]
    assert result["actors"][0]["connection_count"] == 1


def test_daemon_non_member_denied_on_shared_claim(tmp_path: Path) -> None:
    daemon = _make_daemon(tmp_path)
    daemon.svc.set_control_config(require_auth=True)
    token_a = _issue_mgmt_session(daemon, "admin-a", "secret-a")
    token_b = _issue_mgmt_session(daemon, "admin-b", "secret-b")

    claimed = _dispatch(
        daemon,
        seq=1,
        cmd="claim-engine",
        payload={"engine_id": "worker1", "exclusive": False, "session_token": token_a},
    )
    assert claimed["ok"] is True

    denied = _dispatch(
        daemon,
        seq=2,
        cmd="issue-token",
        payload={"engine_id": "worker1", "session_token": token_b},
    )
    assert denied["ok"] is False
    assert denied["error_code"] == "engine_shared_claim_not_member"


def test_daemon_exclusive_owner_conflict_denied(tmp_path: Path) -> None:
    daemon = _make_daemon(tmp_path)
    daemon.svc.set_control_config(require_auth=True)
    token_a = _issue_mgmt_session(daemon, "owner-a", "secret-a")
    token_b = _issue_mgmt_session(daemon, "owner-b", "secret-b")

    first = _dispatch(
        daemon,
        seq=1,
        cmd="claim-engine",
        payload={"engine_id": "worker1", "exclusive": True, "session_token": token_a},
    )
    assert first["ok"] is True

    denied = _dispatch(
        daemon,
        seq=2,
        cmd="claim-engine",
        payload={"engine_id": "worker1", "exclusive": True, "session_token": token_b},
    )
    assert denied["ok"] is False
    assert denied["error_code"] == "exclusive_owner_conflict"


def test_daemon_orphan_takeover_policy(tmp_path: Path) -> None:
    daemon = _make_daemon(tmp_path)
    daemon.svc.set_control_config(
        require_auth=True,
        claim_acl_policy={"owner_ttl_seconds": 10, "audit_event_limit": 200},
    )
    token_a = _issue_mgmt_session(daemon, "owner-a", "secret-a")
    token_b = _issue_mgmt_session(daemon, "owner-b", "secret-b")

    first = _dispatch(
        daemon,
        seq=1,
        cmd="claim-engine",
        payload={"engine_id": "worker1", "exclusive": True, "session_token": token_a},
    )
    assert first["ok"] is True

    keepalive = daemon.svc._owner_keepalive_map(daemon.svc._read_control())  # noqa: SLF001
    keepalive["key:owner-a"] = time.time() - 1200
    control = daemon.svc._read_control()  # noqa: SLF001
    control["claim_owner_keepalive"] = keepalive
    daemon.svc._write_control(control)  # noqa: SLF001

    takeover = _dispatch(
        daemon,
        seq=2,
        cmd="claim-engine",
        payload={"engine_id": "worker1", "exclusive": True, "session_token": token_b},
    )
    assert takeover["ok"] is True
    result = dict(takeover["result"] or {})
    assert result["transition"] == "orphan_takeover"


def test_daemon_localhost_force_override_requires_confirmation(tmp_path: Path) -> None:
    daemon = _make_daemon(tmp_path)
    daemon.svc.set_control_config(require_auth=True)
    token_a = _issue_mgmt_session(daemon, "owner-a", "secret-a")
    token_b = _issue_mgmt_session(daemon, "owner-b", "secret-b")

    first = _dispatch(
        daemon,
        seq=1,
        cmd="claim-engine",
        payload={"engine_id": "worker1", "exclusive": True, "session_token": token_a},
    )
    assert first["ok"] is True

    denied = _dispatch(
        daemon,
        seq=2,
        cmd="claim-engine",
        payload={
            "engine_id": "worker1",
            "exclusive": True,
            "force_override": True,
            "force_override_reason": "policy_recovery",
            "session_token": token_b,
        },
    )
    assert denied["ok"] is False
    assert denied["error_code"] == "localhost_force_override_confirmation_required"

    allowed = _dispatch(
        daemon,
        seq=3,
        cmd="claim-engine",
        payload={
            "engine_id": "worker1",
            "exclusive": True,
            "force_override": True,
            "force_override_confirmation": "CONFIRM_LOCALHOST_FORCE_OVERRIDE",
            "force_override_reason": "policy_recovery",
            "session_token": token_b,
        },
    )
    assert allowed["ok"] is True
    assert dict(allowed["result"] or {}).get("transition") == "force_override"


def test_daemon_force_override_requires_reason(tmp_path: Path) -> None:
    daemon = _make_daemon(tmp_path)
    daemon.svc.set_control_config(require_auth=True)
    token_a = _issue_mgmt_session(daemon, "owner-a", "secret-a")
    token_b = _issue_mgmt_session(daemon, "owner-b", "secret-b")

    first = _dispatch(
        daemon,
        seq=1,
        cmd="claim-engine",
        payload={"engine_id": "worker1", "exclusive": True, "session_token": token_a},
    )
    assert first["ok"] is True

    denied = _dispatch(
        daemon,
        seq=2,
        cmd="claim-engine",
        payload={
            "engine_id": "worker1",
            "exclusive": True,
            "force_override": True,
            "force_override_confirmation": "CONFIRM_LOCALHOST_FORCE_OVERRIDE",
            "session_token": token_b,
        },
    )
    assert denied["ok"] is False
    assert denied["error_code"] == "force_override_reason_required"


def test_daemon_emergency_force_override_allows_without_confirmation_and_audits_high(tmp_path: Path) -> None:
    daemon = _make_daemon(tmp_path)
    daemon.svc.set_control_config(require_auth=True)
    token_a = _issue_mgmt_session(daemon, "owner-a", "secret-a")
    token_b = _issue_mgmt_session(daemon, "owner-b", "secret-b")

    first = _dispatch(
        daemon,
        seq=1,
        cmd="claim-engine",
        payload={"engine_id": "worker1", "exclusive": True, "session_token": token_a},
    )
    assert first["ok"] is True

    allowed = _dispatch(
        daemon,
        seq=2,
        cmd="claim-engine",
        payload={
            "engine_id": "worker1",
            "exclusive": True,
            "force_override": True,
            "force_override_emergency": True,
            "force_override_reason": "owner_malicious",
            "session_token": token_b,
        },
    )
    assert allowed["ok"] is True
    result = dict(allowed.get("result") or {})
    assert str(result.get("transition") or "") == "force_override"
    assert str(result.get("force_override_reason") or "") == "owner_malicious"
    assert bool(result.get("force_override_emergency")) is True

    control = daemon.svc._read_control()  # noqa: SLF001
    rows = list(control.get("claim_audit_events") or [])
    assert rows
    last = dict(rows[-1] or {})
    assert str(last.get("event_type") or "") == "claim_grant"
    assert str(last.get("severity") or "") == "high"
    details = dict(last.get("details") or {})
    assert str(details.get("force_override_reason") or "") == "owner_malicious"
    assert bool(details.get("force_override_emergency")) is True


def test_daemon_emergency_stale_owner_reason_denied_when_owner_still_active(tmp_path: Path) -> None:
    daemon = _make_daemon(tmp_path)
    daemon.svc.set_control_config(require_auth=True)
    token_a = _issue_mgmt_session(daemon, "owner-a", "secret-a")
    token_b = _issue_mgmt_session(daemon, "owner-b", "secret-b")

    first = _dispatch(
        daemon,
        seq=1,
        cmd="claim-engine",
        payload={"engine_id": "worker1", "exclusive": True, "session_token": token_a},
    )
    assert first["ok"] is True

    denied = _dispatch(
        daemon,
        seq=2,
        cmd="claim-engine",
        payload={
            "engine_id": "worker1",
            "exclusive": True,
            "force_override": True,
            "force_override_emergency": True,
            "force_override_reason": "stale_owner_unreachable",
            "session_token": token_b,
        },
    )
    assert denied["ok"] is False
    assert denied["error_code"] == "force_override_emergency_predicate_not_met"
    details = dict(denied.get("error_details") or {})
    assert str(details.get("predicate") or "") == "stale_owner_unreachable_requires_orphan_owner"


def test_daemon_emergency_stale_owner_reason_allowed_when_owner_is_orphan(tmp_path: Path) -> None:
    daemon = _make_daemon(tmp_path)
    daemon.svc.set_control_config(
        require_auth=True,
        claim_acl_policy={"owner_ttl_seconds": 10, "audit_event_limit": 200},
    )
    token_a = _issue_mgmt_session(daemon, "owner-a", "secret-a")
    token_b = _issue_mgmt_session(daemon, "owner-b", "secret-b")

    first = _dispatch(
        daemon,
        seq=1,
        cmd="claim-engine",
        payload={"engine_id": "worker1", "exclusive": True, "session_token": token_a},
    )
    assert first["ok"] is True

    control = daemon.svc._read_control()  # noqa: SLF001
    keepalive = dict(control.get("claim_owner_keepalive") or {})
    keepalive["key:owner-a"] = time.time() - 3600
    control["claim_owner_keepalive"] = keepalive
    daemon.svc._write_control(control)  # noqa: SLF001

    allowed = _dispatch(
        daemon,
        seq=2,
        cmd="claim-engine",
        payload={
            "engine_id": "worker1",
            "exclusive": True,
            "force_override": True,
            "force_override_emergency": True,
            "force_override_reason": "stale_owner_unreachable",
            "session_token": token_b,
        },
    )
    assert allowed["ok"] is True
    result = dict(allowed["result"] or {})
    assert str(result.get("transition") or "") == "orphan_takeover"
    assert bool(result.get("force_override_emergency")) is True


def test_displaced_owner_is_denied_until_reclaim_then_cleared(tmp_path: Path) -> None:
    daemon = _make_daemon(tmp_path)
    daemon.svc.set_control_config(require_auth=True)
    token_a = _issue_mgmt_session(daemon, "owner-a", "secret-a")
    token_b = _issue_mgmt_session(daemon, "owner-b", "secret-b")

    first = _dispatch(
        daemon,
        seq=1,
        cmd="claim-endpoint",
        payload={"exclusive": True, "session_token": token_a},
    )
    assert first["ok"] is True

    takeover = _dispatch(
        daemon,
        seq=2,
        cmd="claim-endpoint",
        payload={
            "exclusive": True,
            "force_override": True,
            "force_override_reason": "policy_recovery",
            "force_override_confirmation": "CONFIRM_LOCALHOST_FORCE_OVERRIDE",
            "session_token": token_b,
        },
    )
    assert takeover["ok"] is True

    denied = _dispatch(
        daemon,
        seq=3,
        cmd="host-metrics",
        payload={"session_token": token_a},
    )
    assert denied["ok"] is False
    assert denied["error_code"] == "ownership_changed_reclaim_required"
    denied_override = _dispatch(
        daemon,
        seq=31,
        cmd="set-endpoint-mode-override",
        payload={"mode": "shared", "session_token": token_a},
    )
    assert denied_override["ok"] is False
    assert denied_override["error_code"] == "ownership_changed_reclaim_required"
    denied_effective = _dispatch(
        daemon,
        seq=32,
        cmd="get-endpoint-mode-effective",
        payload={"session_token": token_a},
    )
    assert denied_effective["ok"] is False
    assert denied_effective["error_code"] == "ownership_changed_reclaim_required"

    reclaim = _dispatch(
        daemon,
        seq=4,
        cmd="claim-endpoint",
        payload={
            "exclusive": True,
            "force_override": True,
            "force_override_reason": "policy_recovery",
            "force_override_confirmation": "CONFIRM_LOCALHOST_FORCE_OVERRIDE",
            "session_token": token_a,
        },
    )
    assert reclaim["ok"] is True

    allowed = _dispatch(
        daemon,
        seq=5,
        cmd="host-metrics",
        payload={"session_token": token_a},
    )
    assert allowed["ok"] is True
    allowed_auth = dict(dict(allowed.get("result") or {}).get("auth_status") or {})
    assert allowed_auth["caller_key_id"] == "owner-a"
    assert allowed_auth["caller_role"] == "admin"


def test_daemon_non_localhost_shared_claim_denied(tmp_path: Path) -> None:
    daemon = _make_daemon(tmp_path)
    daemon.svc.set_control_config(require_auth=True)
    token = _issue_mgmt_session(daemon, "admin-x", "secret-x")

    denied = _dispatch(
        daemon,
        seq=1,
        cmd="claim-engine",
        payload={"engine_id": "worker1", "exclusive": False, "session_token": token},
        peer_host="10.20.30.40",
    )
    assert denied["ok"] is False
    assert denied["error_code"] == "non_localhost_shared_claim_denied"


def test_daemon_no_auth_forces_exclusive_claim_even_if_shared_requested(tmp_path: Path) -> None:
    daemon = _make_daemon(tmp_path)
    daemon.svc.set_control_config(
        require_auth=False,
        access_profile={"connectivity_mode": "local_only"},
        endpoint_mode_default="exclusive",
    )

    out = _dispatch(
        daemon,
        seq=1,
        cmd="claim-engine",
        payload={"engine_id": "worker1", "exclusive": False, "backend_id": "local-client"},
        peer_host="127.0.0.1",
    )
    assert out["ok"] is True
    result = dict(out.get("result") or {})
    assert str(result.get("mode") or "") == "exclusive"


def test_daemon_operation_start_and_status(tmp_path: Path) -> None:
    daemon = _make_daemon(tmp_path)
    daemon.svc.set_control_config(require_auth=True)
    token = _issue_mgmt_session(daemon, "admin-op", "secret-op")

    started = _dispatch(
        daemon,
        seq=1,
        cmd="op-start",
        payload={
            "command": "discover-running",
            "payload": {"session_token": token, "include_progress": True},
        },
    )
    assert started["ok"] is True
    op = dict(started.get("result") or {})
    op_id = str(op.get("operation_id") or "")
    assert op_id

    status = _dispatch(
        daemon,
        seq=2,
        cmd="op-status",
        payload={"operation_id": op_id, "session_token": token},
    )
    assert status["ok"] is True
    status_result = dict(status.get("result") or {})
    assert str(status_result.get("operation_id") or "") == op_id
    assert isinstance(status_result.get("progress_events"), list)

    denied = _dispatch(
        daemon,
        seq=3,
        cmd="op-status",
        payload={"operation_id": op_id},
    )
    assert denied["ok"] is False
    assert denied["error_code"] == "missing_or_invalid_session_token"


def test_daemon_operation_start_copies_outer_session_token(tmp_path: Path) -> None:
    daemon = _make_daemon(tmp_path)
    daemon.svc.set_control_config(require_auth=True)
    token = _issue_mgmt_session(daemon, "admin-op", "secret-op")

    started = _dispatch(
        daemon,
        seq=1,
        cmd="op-start",
        payload={
            "command": "discover-running",
            "payload": {"include_progress": True},
            "session_token": token,
        },
    )

    assert started["ok"] is True
    op_id = str((started.get("result") or {}).get("operation_id") or "")
    assert op_id
    status = _dispatch(
        daemon,
        seq=2,
        cmd="op-status",
        payload={"operation_id": op_id, "session_token": token},
    )
    assert status["ok"] is True


def test_connect_operation_status_reports_worker_ready_wait(tmp_path: Path) -> None:
    daemon = _make_daemon(tmp_path)

    def _slow_call_service(cmd: str, payload: dict) -> dict:
        assert cmd == "connect-from-config"
        time.sleep(0.2)
        return {"status": "ok", "stage": "completed", "progress_events": []}

    daemon._call_service = _slow_call_service  # type: ignore[method-assign]

    async def _run() -> None:
        started = await _dispatch_async(
            daemon,
            seq=1,
            cmd="op-start",
            payload={"command": "connect-from-config", "payload": {"config_path": "default"}},
        )
        assert started["ok"] is True
        op_id = str((started.get("result") or {}).get("operation_id") or "")
        assert op_id

        found = False
        for seq in range(2, 12):
            status = await _dispatch_async(
                daemon,
                seq=seq,
                cmd="op-status",
                payload={"operation_id": op_id},
            )
            assert status["ok"] is True
            events = list((status.get("result") or {}).get("progress_events") or [])
            found = any(
                str(x.get("stage") or "") == "connect.worker_ready" and str(x.get("status") or "") == "running"
                for x in events
            )
            if found:
                break
            await asyncio.sleep(0.03)
        assert found

    asyncio.run(_run())


def test_connect_operation_status_reports_log_weight_progress(tmp_path: Path) -> None:
    daemon = _make_daemon(tmp_path)
    log_path = tmp_path / "logs" / "model.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        "Loading weights:  74%|#######4  | 268/362 [00:03<00:01]\n",
        encoding="utf-8",
    )
    daemon.svc.engines_state_file.write_text(
        json.dumps(
            {
                "version": 1,
                "engines": [
                    {
                        "engine_id": "model-granite",
                        "model_instance_id": "model-granite",
                        "canonical_model_path": str((tmp_path / "granite").resolve()),
                        "canonical_config_path": str((tmp_path / "config.json").resolve()),
                        "log_path": str(log_path),
                        "spawned_at": time.time(),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    op = daemon._create_operation(  # noqa: SLF001
        command="connect-from-config",
        payload={"config_path": str(tmp_path / "config.json"), "model_path": str(tmp_path / "granite")},
    )
    op_id = str(op.get("operation_id") or "")

    status = _dispatch(
        daemon,
        seq=1,
        cmd="op-status",
        payload={"operation_id": op_id},
    )

    assert status["ok"] is True
    result = dict(status.get("result") or {})
    assert result.get("progress_percent") == 74
    assert "Loading model weights" in str(result.get("progress_text") or "")
    diagnostics = dict(result.get("diagnostics") or {})
    worker_log = dict(diagnostics.get("worker_log") or {})
    assert worker_log.get("log_path") == str(log_path)


def test_daemon_operation_status_survives_memory_reload(tmp_path: Path) -> None:
    daemon = _make_daemon(tmp_path)
    op = daemon._create_operation(command="discover-running", payload={})  # noqa: SLF001
    op_id = str(op.get("operation_id") or "")

    reloaded = _make_daemon(tmp_path)
    status = _dispatch(
        reloaded,
        seq=1,
        cmd="op-status",
        payload={"operation_id": op_id},
    )

    assert status["ok"] is True
    result = dict(status.get("result") or {})
    assert result.get("operation_id") == op_id
    assert (tmp_path / "state" / "operations.json").exists()
    assert (tmp_path / "state" / "operation_audit.jsonl").exists()


def test_daemon_operation_cancel_marks_running_task_canceled(tmp_path: Path) -> None:
    daemon = _make_daemon(tmp_path)

    def _slow_call_service(cmd: str, payload: dict) -> dict:
        time.sleep(0.2)
        return {"status": "ok", "command": cmd, "payload": payload}

    daemon._call_service = _slow_call_service  # type: ignore[method-assign]

    async def _run() -> None:
        started = await _dispatch_async(
            daemon,
            seq=1,
            cmd="op-start",
            payload={"command": "discover-running", "payload": {}},
        )
        assert started["ok"] is True
        op_id = str((started.get("result") or {}).get("operation_id") or "")
        assert op_id

        canceled = await _dispatch_async(
            daemon,
            seq=2,
            cmd="op-cancel",
            payload={"operation_id": op_id, "reason": "test_cancel"},
        )
        assert canceled["ok"] is True
        cancel_result = dict(canceled.get("result") or {})
        assert cancel_result.get("cancel_requested") is True
        assert str(cancel_result.get("cancel_status") or "") in {"cancel_requested", "canceled"}

        status = await _dispatch_async(
            daemon,
            seq=3,
            cmd="op-status",
            payload={"operation_id": op_id},
        )
        assert status["ok"] is True
        status_result = dict(status.get("result") or {})
        assert status_result.get("done") is True
        assert status_result.get("status") == "canceled"
        assert status_result.get("cancel_requested") is True

    asyncio.run(_run())


def test_daemon_operation_cancel_requires_operation_session_token(tmp_path: Path) -> None:
    daemon = _make_daemon(tmp_path)
    daemon.svc.set_control_config(require_auth=True)
    token = _issue_mgmt_session(daemon, "admin-cancel", "secret-cancel")

    def _slow_call_service(cmd: str, payload: dict) -> dict:
        time.sleep(0.2)
        return {"status": "ok", "command": cmd}

    daemon._call_service = _slow_call_service  # type: ignore[method-assign]

    async def _run() -> None:
        started = await _dispatch_async(
            daemon,
            seq=1,
            cmd="op-start",
            payload={
                "command": "discover-running",
                "payload": {"session_token": token},
            },
        )
        assert started["ok"] is True
        op_id = str((started.get("result") or {}).get("operation_id") or "")
        assert op_id

        denied = await _dispatch_async(
            daemon,
            seq=2,
            cmd="op-cancel",
            payload={"operation_id": op_id},
        )
        assert denied["ok"] is False
        assert denied["error_code"] == "missing_or_invalid_session_token"

        allowed = await _dispatch_async(
            daemon,
            seq=3,
            cmd="op-cancel",
            payload={"operation_id": op_id, "session_token": token},
        )
        assert allowed["ok"] is True

    asyncio.run(_run())


def test_daemon_operation_cancel_tears_down_known_connect_engine(tmp_path: Path) -> None:
    daemon = _make_daemon(tmp_path)
    op = daemon._create_operation(  # noqa: SLF001
        command="connect-from-config",
        payload={"config_path": "default", "engine_id": "worker-cancel"},
    )
    op_id = str(op.get("operation_id") or "")
    calls: list[str] = []

    def _shutdown(engine_id: str, timeout_seconds: float = 8.0) -> dict:
        calls.append(engine_id)
        return {"status": "stopped", "engine_id": engine_id, "timeout_seconds": timeout_seconds}

    daemon.svc.shutdown = _shutdown  # type: ignore[method-assign]

    canceled = _dispatch(
        daemon,
        seq=1,
        cmd="op-cancel",
        payload={"operation_id": op_id},
    )
    assert canceled["ok"] is True
    assert calls == ["worker-cancel"]
    result = dict(canceled.get("result") or {})
    assert result.get("status") == "canceled"
    assert result.get("cancel_teardown_attempted") is True
    assert result.get("cancel_teardown_status") == "stopped"


def test_daemon_operation_cancel_tears_down_late_connect_engine_id(tmp_path: Path) -> None:
    daemon = _make_daemon(tmp_path)
    calls: list[str] = []

    def _connect_call_service(cmd: str, payload: dict) -> dict:
        time.sleep(0.05)
        assert cmd == "connect-from-config"
        return {"status": "ok", "engine_id": "late-worker-cancel"}

    def _shutdown(engine_id: str, timeout_seconds: float = 8.0) -> dict:
        calls.append(engine_id)
        return {"status": "stopped", "engine_id": engine_id, "timeout_seconds": timeout_seconds}

    daemon._call_service = _connect_call_service  # type: ignore[method-assign]
    daemon.svc.shutdown = _shutdown  # type: ignore[method-assign]

    async def _run() -> None:
        started = await _dispatch_async(
            daemon,
            seq=1,
            cmd="op-start",
            payload={"command": "connect-from-config", "payload": {"config_path": "default"}},
        )
        assert started["ok"] is True
        op_id = str((started.get("result") or {}).get("operation_id") or "")
        assert op_id

        canceled = await _dispatch_async(
            daemon,
            seq=2,
            cmd="op-cancel",
            payload={"operation_id": op_id},
        )
        assert canceled["ok"] is True

        await asyncio.sleep(0.1)
        status = await _dispatch_async(
            daemon,
            seq=3,
            cmd="op-status",
            payload={"operation_id": op_id},
        )
        result = dict(status.get("result") or {})
        assert result.get("status") == "canceled"
        assert result.get("target_engine_id") == "late-worker-cancel"
        assert result.get("cancel_teardown_status") == "stopped"
        assert calls == ["late-worker-cancel"]

    asyncio.run(_run())


def test_owner_disconnect_shutdown_policy_sets_stop_event_for_exclusive_owner(tmp_path: Path) -> None:
    daemon = _make_daemon(tmp_path)
    daemon.svc.set_control_config(
        require_auth=True,
        lifecycle_profile="detached_user_process",
        lifecycle_policy={"owner_disconnect_shutdown": True},
    )
    token = _issue_mgmt_session(daemon, "owner-main", "secret-main")
    claimed = _dispatch(
        daemon,
        seq=1,
        cmd="claim-endpoint",
        payload={"exclusive": True, "session_token": token},
    )
    assert claimed["ok"] is True
    daemon._stop_event = asyncio.Event()  # noqa: SLF001
    daemon._track_actor_connected("key:owner-main")  # noqa: SLF001
    did_shutdown = daemon._apply_owner_disconnect_policy({"key:owner-main"})  # noqa: SLF001
    assert did_shutdown is True
    assert daemon._stop_event.is_set() is True  # noqa: SLF001


def test_owner_disconnect_exclusive_owner_still_stops_daemon_when_policy_disabled(tmp_path: Path) -> None:
    daemon = _make_daemon(tmp_path)
    daemon.svc.set_control_config(
        require_auth=True,
        lifecycle_profile="detached_user_process",
        lifecycle_policy={"owner_disconnect_shutdown": False},
    )
    token = _issue_mgmt_session(daemon, "owner-main", "secret-main")
    claimed = _dispatch(
        daemon,
        seq=1,
        cmd="claim-endpoint",
        payload={"exclusive": True, "session_token": token},
    )
    assert claimed["ok"] is True
    daemon._stop_event = asyncio.Event()  # noqa: SLF001
    daemon._track_actor_connected("key:owner-main")  # noqa: SLF001
    did_shutdown = daemon._apply_owner_disconnect_policy({"key:owner-main"})  # noqa: SLF001
    assert did_shutdown is True
    assert daemon._stop_event.is_set() is True  # noqa: SLF001


def test_daemon_terminal_control_disabled_blocks_shutdown_token_path(tmp_path: Path) -> None:
    daemon = _make_daemon(tmp_path)
    daemon.svc.set_control_config(
        require_auth=True,
        lifecycle_profile="service_managed",
        lifecycle_policy={"terminal_control_enabled": False},
    )
    daemon._stop_event = asyncio.Event()  # noqa: SLF001
    out = _dispatch(
        daemon,
        seq=1,
        cmd="__shutdown__",
        payload={"shutdown_token": daemon.shutdown_token},
    )
    assert out["ok"] is False
    assert out["error_code"] == "terminal_control_disabled"
    assert daemon._stop_event.is_set() is False  # noqa: SLF001


def test_daemon_terminal_control_disabled_blocks_endpoint_mode_override(tmp_path: Path) -> None:
    daemon = _make_daemon(tmp_path)
    daemon.svc.set_control_config(
        require_auth=True,
        lifecycle_profile="service_managed",
        lifecycle_policy={"terminal_control_enabled": False},
    )
    token = _issue_mgmt_session(daemon, "admin-override", "secret-override")
    out = _dispatch(
        daemon,
        seq=1,
        cmd="set-endpoint-mode-override",
        payload={"mode": "exclusive", "session_token": token},
    )
    assert out["ok"] is False
    assert out["error_code"] == "terminal_control_disabled"


def test_daemon_endpoint_mode_override_requires_auth_token(tmp_path: Path) -> None:
    daemon = _make_daemon(tmp_path)
    daemon.svc.set_control_config(require_auth=True)
    out = _dispatch(
        daemon,
        seq=1,
        cmd="set-endpoint-mode-override",
        payload={"mode": "exclusive"},
    )
    assert out["ok"] is False
    assert out["error"] == "auth_failed"
    assert out["error_code"] == "session_token_required"
