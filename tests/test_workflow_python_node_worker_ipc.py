from __future__ import annotations

from hosting.workflow_python_node_worker_ipc import HostApi, SandboxApi


def test_python_node_host_grouped_aliases_call_canonical_methods() -> None:
    host = HostApi(conn=object(), request_id="req-alias")
    calls: list[tuple[str, dict]] = []

    def _call(method: str, arguments: dict | None = None) -> dict:
        calls.append((method, dict(arguments or {})))
        return {"method": method}

    host.call = _call  # type: ignore[method-assign]

    assert host.fs.read_text("in", "a.txt") == {"method": "fs.read_text"}
    assert host.fs.write_text("out", "b.txt", "hello") == {"method": "fs.write_text"}
    assert host.fs.list("in") == {"method": "fs.list"}
    assert host.fs.stat("in", "a.txt") == {"method": "fs.stat"}
    assert host.fs.mkdir("out", "nested") == {"method": "fs.mkdir"}
    assert host.http.fetch("https://example.com/api") == {"method": "http.fetch"}

    assert calls == [
        ("fs.read_text", {"root_id": "in", "relative_path": "a.txt", "encoding": "utf-8"}),
        (
            "fs.write_text",
            {
                "root_id": "out",
                "relative_path": "b.txt",
                "text": "hello",
                "encoding": "utf-8",
                "create_parents": True,
            },
        ),
        ("fs.list", {"root_id": "in", "relative_path": ""}),
        ("fs.stat", {"root_id": "in", "relative_path": "a.txt"}),
        ("fs.mkdir", {"root_id": "out", "relative_path": "nested", "parents": True, "exist_ok": True}),
        (
            "http.fetch",
            {
                "url": "https://example.com/api",
                "method": "GET",
                "headers": {},
                "body_b64": "",
                "timeout_seconds": 30.0,
                "max_response_bytes": 1024 * 1024,
            },
        ),
    ]


def test_python_node_discovery_surfaces_keep_distinct_canonical_methods() -> None:
    host = HostApi(conn=object(), request_id="req-discovery")
    sandbox = SandboxApi(host)
    calls: list[tuple[str, dict]] = []

    def _call(method: str, arguments: dict | None = None) -> dict:
        calls.append((method, dict(arguments or {})))
        if method == "host.describe":
            return {"contract": "hosting.sandbox.host_capabilities.v1"}
        if method == "sandbox.describe":
            return {"contract": "hosting.sandbox.discovery.v1"}
        raise AssertionError(method)

    host.call = _call  # type: ignore[method-assign]

    assert host.describe()["contract"] == "hosting.sandbox.host_capabilities.v1"
    assert sandbox.describe()["contract"] == "hosting.sandbox.discovery.v1"
    assert calls == [("host.describe", {}), ("sandbox.describe", {})]
