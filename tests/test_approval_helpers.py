from __future__ import annotations

from pathlib import Path

from hosting import host_capability_approval_check_fs_path, host_capability_approval_check_http_fetch


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
