from __future__ import annotations

import hashlib

from hosting.sandbox.workflow_js_bundle import (
    build_workflow_js_bundle,
    build_workflow_js_bundle_request,
    workflow_js_host_bridge_imports,
)


def test_workflow_js_bundle_rewrites_allowed_host_bridge_imports() -> None:
    source = """
import fs, { readText, writeText as write } from "@host/fs";
import * as crypto from "@host/crypto";
import { base64Encode } from "@host/codec";

exports.run = function(input) {
  const text = readText("seed", "");
  return {output: {hash: crypto.sha256(base64Encode(text)), fs: fs, write: write}};
};
"""

    bundle = build_workflow_js_bundle(source, host_description={"methods": ["host.describe", "fs.read_text", "fs.write_text"]})

    assert bundle["ok"] is True
    assert bundle["resolved_allowed_imports"] == ["@host/codec", "@host/crypto", "@host/fs"]
    assert bundle["resolved_disabled_imports"] == []
    assert bundle["unresolved_imports"] == []
    assert 'from "@host/fs"' not in bundle["module_source"]
    assert "const fs = api.fs;" in bundle["module_source"]
    assert "const { readText, writeText: write } = api.fs;" in bundle["module_source"]
    assert "const crypto = api.crypto;" in bundle["module_source"]
    assert "const { base64Encode } = api.codec;" in bundle["module_source"]
    assert bundle["module_sha256"] == hashlib.sha256(bundle["module_source"].encode("utf-8")).hexdigest()


def test_workflow_js_bundle_reports_disabled_host_bridge_imports() -> None:
    source = 'import { fetch } from "@host/http";\nexports.run = function() { return fetch("https://example.test"); };\n'

    bundle = build_workflow_js_bundle(source)

    assert bundle["ok"] is False
    assert bundle["resolved_allowed_imports"] == []
    assert bundle["resolved_disabled_imports"] == ["@host/http"]
    assert bundle["unresolved_imports"] == []
    assert 'import { fetch } from "@host/http";' in bundle["module_source"]


def test_workflow_js_bundle_allows_http_when_host_description_exposes_it() -> None:
    source = 'import { fetch } from "@host/http";\nexports.run = function() { return fetch("https://example.test"); };\n'

    bundle = build_workflow_js_bundle(source, host_description={"methods": ["host.describe", "http.fetch"]})

    assert bundle["ok"] is True
    assert bundle["resolved_allowed_imports"] == ["@host/http"]
    assert "const { fetch } = api.http;" in bundle["module_source"]


def test_workflow_js_bundle_reports_unresolved_imports() -> None:
    source = 'import leftPad from "left-pad";\nexports.run = function() { return leftPad("x", 2); };\n'

    bundle = build_workflow_js_bundle(source)

    assert bundle["ok"] is False
    assert bundle["resolved_allowed_imports"] == []
    assert bundle["resolved_disabled_imports"] == []
    assert bundle["unresolved_imports"] == ["left-pad"]
    assert 'import leftPad from "left-pad";' in bundle["module_source"]


def test_workflow_js_bundle_supports_side_effect_bridge_imports() -> None:
    source = 'import "@host/api";\nexports.run = function(input) { return {output: input}; };\n'

    bundle = build_workflow_js_bundle(source)

    assert bundle["ok"] is True
    assert bundle["resolved_allowed_imports"] == ["@host/api"]
    assert 'import "@host/api"' not in bundle["module_source"]
    assert "workflow-js-bundle host bridge import: @host/api" in bundle["module_source"]


def test_workflow_js_bundle_accepts_custom_bridge_table() -> None:
    source = 'import tool from "@tool/demo";\nexports.run = function() { return tool.echo("ok"); };\n'

    bundle = build_workflow_js_bundle(
        source,
        bridge_imports={
            "@tool/demo": {
                "expression": "api.call.bind(null, 'tool.demo')",
                "enabled": True,
            }
        },
    )

    assert bundle["ok"] is True
    assert bundle["resolved_allowed_imports"] == ["@tool/demo"]
    assert "const tool = api.call.bind(null, 'tool.demo');" in bundle["module_source"]


def test_workflow_js_bundle_request_embeds_bundle_diagnostics() -> None:
    source = 'import { describe } from "@host/api";\nexports.run = function() { return {output: describe()}; };\n'

    request = build_workflow_js_bundle_request(
        source,
        package_id="pkg",
        workflow_id="wf",
        package_source_digest="sha256:source",
        payload={"value": 1},
    )

    assert request["package_id"] == "pkg"
    assert request["workflow_id"] == "wf"
    assert request["payload"] == {"value": 1}
    assert request["module_sha256"] == hashlib.sha256(request["module_source"].encode("utf-8")).hexdigest()
    assert request["javascript"]["bundle"]["ok"] is True
    assert request["javascript"]["bundle"]["resolved_allowed_imports"] == ["@host/api"]


def test_workflow_js_host_bridge_imports_respects_sandbox_http_policy() -> None:
    bridges = workflow_js_host_bridge_imports(
        sandbox_policy={
            "sandbox": {
                "enabled": True,
                "brokered_io": {"http": True},
                "network": {"mode": "brokered_only"},
                "host_api": {"namespaces": {"http": True}},
            }
        }
    )

    assert bridges["@host/http"].enabled is True

    disabled = workflow_js_host_bridge_imports(sandbox_policy={"sandbox": {"host_api": {"namespaces": {"fs": False}}}})

    assert disabled["@host/fs"].enabled is False
