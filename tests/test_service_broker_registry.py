from __future__ import annotations

from hosting.sandbox.host_api import known_host_capability_method_descriptors
from hosting.sandbox.service_broker_registry import (
    SERVICE_BROKER_CONTRACT,
    SERVICE_BROKER_PROVIDER_ID,
    SERVICE_BROKER_PROVIDER_KIND,
    service_broker_discover,
    service_broker_method_descriptors,
    service_broker_method_policy_hint,
)


def test_service_broker_registry_derives_descriptors_from_docstrings() -> None:
    descriptors = {row["name"]: row for row in service_broker_method_descriptors()}
    read_text = descriptors["fs.read_text"]

    assert read_text["provider"] == {
        "provider_id": SERVICE_BROKER_PROVIDER_ID,
        "kind": SERVICE_BROKER_PROVIDER_KIND,
        "owner": "service",
        "visibility": "request",
    }
    assert "Read UTF text" in read_text["description"]
    assert read_text["args_schema"]["required"] == ["root_id", "relative_path"]
    assert read_text["args_schema"]["properties"]["root_id"]["description"].startswith("Brokered filesystem root")
    assert read_text["args_schema"]["properties"]["encoding"]["default"] == "utf-8"
    assert read_text["permissions"] == ["artifact.read"]
    assert read_text["metadata"]["service_broker"]["contract"] == SERVICE_BROKER_CONTRACT
    assert read_text["metadata"]["service_broker"]["policy_hint"] == {
        "kind": "filesystem",
        "access": "read",
        "allow_empty_relative_path": False,
    }
    assert read_text["metadata"]["concurrency"] == {
        "mode": "parallel",
        "group": "filesystem",
        "max_concurrency": 32,
        "queue_policy": "bounded",
        "queue_depth": 64,
        "queue_timeout_seconds": 30.0,
        "thread_safe_required": True,
    }


def test_service_broker_discovery_returns_contract_descriptions() -> None:
    discovery = service_broker_discover(include_fs=True, include_http=False)

    assert discovery["contract"] == SERVICE_BROKER_CONTRACT
    assert discovery["provider"]["kind"] == SERVICE_BROKER_PROVIDER_KIND
    assert "fs.read_text" in discovery["method_names"]
    assert "http.fetch" not in discovery["method_names"]
    assert all(row["contract"] == SERVICE_BROKER_CONTRACT for row in discovery["methods"])
    assert {row["name"]: row["policy_hint"] for row in discovery["methods"]}["fs.list"] == {
        "kind": "filesystem",
        "access": "read",
        "allow_empty_relative_path": True,
    }


def test_service_broker_policy_hints_are_owned_by_registry() -> None:
    assert service_broker_method_policy_hint("fs.write_text") == {
        "kind": "filesystem",
        "access": "write",
        "allow_empty_relative_path": False,
    }
    assert service_broker_method_policy_hint("http.fetch") == {"kind": "http", "operation": "fetch"}
    assert service_broker_method_policy_hint("unknown") == {}


def test_known_host_capability_descriptors_delegate_to_service_broker_registry() -> None:
    known = known_host_capability_method_descriptors(include_fs=False, include_http=True)

    assert [row["name"] for row in known] == ["http.fetch"]
    assert known[0]["provider"]["kind"] == SERVICE_BROKER_PROVIDER_KIND
