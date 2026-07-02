from __future__ import annotations

from pathlib import Path

from hosting import (
    host_capability_approval_check_fs_path,
    host_capability_approval_check_http_fetch,
    host_capability_approval_check_service_broker_request,
    service_broker_method_policy_hint,
)


def test_host_capability_approval_check_fs_path_allows_policy_root(tmp_path: Path) -> None:
    policy = {
        "sandbox": {
            "enabled": True,
            "filesystem": {"rules": [{"root_id": "project", "path": str(tmp_path), "access": ["read"]}]},
            "brokered_io": {"filesystem": True},
        }
    }
    request = {"argument_preview": {"root_id": "project", "relative_path": "src/app.py"}}

    out = host_capability_approval_check_fs_path(request, policy)

    assert out["allowed"] is True
    assert out["relative_path"] == "src/app.py"
    assert out["root_path"] == str(tmp_path.resolve())


def test_host_capability_approval_check_fs_path_denies_escape(tmp_path: Path) -> None:
    policy = {
        "sandbox": {
            "enabled": True,
            "filesystem": {"rules": [{"root_id": "project", "path": str(tmp_path), "access": ["read"]}]},
            "brokered_io": {"filesystem": True},
        }
    }
    request = {"argument_preview": {"root_id": "project", "relative_path": "../secret.txt"}}

    out = host_capability_approval_check_fs_path(request, policy)

    assert out["allowed"] is False
    assert out["reason"] == "path_traversal_denied"


def test_host_capability_approval_check_fs_path_denies_outside_scope(tmp_path: Path) -> None:
    policy = {
        "sandbox": {
            "enabled": True,
            "filesystem": {"rules": [{"root_id": "project", "path": str(tmp_path), "access": ["read"]}]},
            "brokered_io": {"filesystem": True},
        }
    }
    request = {"argument_preview": {"root_id": "project", "relative_path": "tests/test_app.py"}}

    out = host_capability_approval_check_fs_path(request, policy, scoped_root="src")

    assert out["allowed"] is False
    assert out["reason"] == "outside_approved_scope"


def test_host_capability_approval_check_http_fetch_validates_policy() -> None:
    policy = {
        "sandbox": {
            "enabled": True,
            "brokered_io": {"http": True},
            "network": {"mode": "brokered_only", "allow_url_prefixes": ["https://example.com/api/"]},
        }
    }

    allowed = host_capability_approval_check_http_fetch(
        {"argument_preview": {"url": "https://example.com/api/status", "method": "GET"}},
        policy,
    )
    denied = host_capability_approval_check_http_fetch(
        {"argument_preview": {"url": "https://example.com/admin", "method": "GET"}},
        policy,
    )

    assert allowed["allowed"] is True
    assert denied["allowed"] is False
    assert denied["reason"] == "brokered_http_url_not_allowed"


def test_service_broker_method_policy_hint_is_registry_backed() -> None:
    assert service_broker_method_policy_hint("fs.read_text") == {
        "kind": "filesystem",
        "access": "read",
        "allow_empty_relative_path": False,
    }
    assert service_broker_method_policy_hint("fs.list") == {
        "kind": "filesystem",
        "access": "read",
        "allow_empty_relative_path": True,
    }
    assert service_broker_method_policy_hint("http.fetch") == {"kind": "http", "operation": "fetch"}
    assert service_broker_method_policy_hint("missing.method") == {}


def test_host_capability_approval_check_service_broker_request_dispatches_fs(tmp_path: Path) -> None:
    policy = {
        "sandbox": {
            "enabled": True,
            "filesystem": {"rules": [{"root_id": "project", "path": str(tmp_path), "access": ["read"]}]},
            "brokered_io": {"filesystem": True},
        }
    }

    allowed = host_capability_approval_check_service_broker_request(
        {"method": "fs.list", "argument_preview": {"root_id": "project", "relative_path": ""}},
        policy,
    )
    denied = host_capability_approval_check_service_broker_request(
        {"method": "fs.write_text", "argument_preview": {"root_id": "project", "relative_path": "out.txt"}},
        policy,
    )

    assert allowed["allowed"] is True
    assert allowed["relative_path"] == "."
    assert denied["allowed"] is False
    assert denied["reason"] == "fs_access_denied"


def test_host_capability_approval_check_service_broker_request_dispatches_http() -> None:
    policy = {
        "sandbox": {
            "enabled": True,
            "brokered_io": {"http": True},
            "network": {"mode": "brokered_only", "allow_hosts": ["example.com"]},
        }
    }

    out = host_capability_approval_check_service_broker_request(
        {"method": "http.fetch", "argument_preview": {"url": "https://example.com/", "method": "GET"}},
        policy,
    )

    assert out["allowed"] is True
    assert out["host"] == "example.com"


def test_host_capability_approval_check_service_broker_request_denies_unknown() -> None:
    out = host_capability_approval_check_service_broker_request({"method": "state.read", "argument_preview": {}}, {})

    assert out["allowed"] is False
    assert out["reason"] == "unsupported_service_broker_method:state.read"
