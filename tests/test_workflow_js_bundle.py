from __future__ import annotations

import hashlib

from hosting.sandbox.workflow_js_bundle import (
    build_workflow_js_bundle,
    build_workflow_js_module_bundle,
    build_workflow_js_bundle_request,
    describe_workflow_js_bundle_source,
    extract_workflow_js_bundle_segment,
    resolve_workflow_js_bundle_line,
    workflow_js_host_bridge_imports,
)
from hosting.sandbox.workflow_js_node_runtime import WorkflowJsNodeRuntimeRegistry


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


def test_workflow_js_module_bundle_inlines_passed_modules_and_host_bridges() -> None:
    bundle = build_workflow_js_module_bundle(
        entry_module="main.js",
        host_description={"methods": ["host.describe"]},
        modules=[
            {
                "id": "main.js",
                "source": (
                    'import { readSeed } from "./lib/io.js";\n'
                    'import { sha256 } from "@host/crypto";\n'
                    "export function run(input) {\n"
                    "  return {output: {digest: sha256(readSeed(input.value))}};\n"
                    "}\n"
                ),
            },
            {
                "id": "lib/io.js",
                "source": "export function readSeed(value) { return 'seed:' + value; }\n",
            },
        ],
    )

    assert bundle["ok"] is True
    assert bundle["resolved_modules"] == ["lib/io.js", "main.js"]
    assert bundle["resolved_allowed_imports"] == ["./lib/io.js", "@host/crypto"]
    assert bundle["resolved_disabled_imports"] == []
    assert bundle["unresolved_imports"] == []
    assert bundle["rejected_imports"] == []
    assert bundle["module_sha256"] == hashlib.sha256(bundle["module_source"].encode("utf-8")).hexdigest()

    out = WorkflowJsNodeRuntimeRegistry().execute(
        {
            "request_id": "req-module-bundle",
            "module_source": bundle["module_source"],
            "module_sha256": bundle["module_sha256"],
            "package_id": "pkg",
            "workflow_id": "wf",
            "package_source_digest": "digest",
            "payload": {"value": 7},
            "limits": {"timeout_ms": 5000, "output_limit_bytes": 65536, "memory_limit_mb": 128},
        }
    )

    assert out["ok"] is True
    assert out["output"] == {"digest": hashlib.sha256(b"seed:7").hexdigest()}


def test_workflow_js_module_bundle_marks_segments_and_resolves_lines() -> None:
    bundle = build_workflow_js_module_bundle(
        entry_module="main.js",
        modules=[
            {
                "id": "main.js",
                "source": (
                    'import { value } from "./value.js";\n'
                    "export function run(input) {\n"
                    "  const total = value + input.add;\n"
                    "  return {output: total};\n"
                    "}\n"
                ),
            },
            {"id": "value.js", "source": "export const value = 40;\n"},
        ],
    )

    segments = describe_workflow_js_bundle_source(bundle["module_source"])
    names = {segment["name"] for segment in segments}
    main_segment = extract_workflow_js_bundle_segment(bundle, "main.js")
    runtime_segment = extract_workflow_js_bundle_segment(bundle, "runtime:prelude")
    generated_line = next(
        index
        for index, line in enumerate(bundle["module_source"].splitlines(), start=1)
        if "const total = value + input.add;" in line
    )
    resolved = resolve_workflow_js_bundle_line(bundle, generated_line)

    assert {"runtime:prelude", "runtime:entry", "main.js", "value.js"}.issubset(names)
    assert main_segment is not None
    assert "const total = value + input.add;" in main_segment
    assert "__workflowJsDefine" not in main_segment
    assert runtime_segment is not None
    assert "function __workflowJsRequire" in runtime_segment
    assert resolved["module"] == "main.js"
    assert resolved["original_line"] == 3


def test_workflow_js_module_bundle_reads_missing_relative_modules_from_local_root(tmp_path) -> None:
    (tmp_path / "lib").mkdir()
    (tmp_path / "main.js").write_text('import answer from "./lib/answer.js";\nexport function run() { return {output: answer}; }\n', encoding="utf-8")
    (tmp_path / "lib" / "answer.js").write_text("export default 42;\n", encoding="utf-8")

    bundle = build_workflow_js_module_bundle(entry_module="main.js", local_roots=[tmp_path])

    assert bundle["ok"] is True
    assert bundle["resolved_modules"] == ["lib/answer.js", "main.js"]
    assert bundle["unresolved_imports"] == []

    out = WorkflowJsNodeRuntimeRegistry().execute(
        {
            "request_id": "req-module-bundle-local-root",
            "module_source": bundle["module_source"],
            "module_sha256": bundle["module_sha256"],
            "package_id": "pkg",
            "workflow_id": "wf",
            "package_source_digest": "digest",
            "payload": {},
            "limits": {"timeout_ms": 5000, "output_limit_bytes": 65536, "memory_limit_mb": 128},
        }
    )

    assert out["ok"] is True
    assert out["output"] == 42


def test_workflow_js_module_bundle_resolves_extensionless_mjs_from_local_root(tmp_path) -> None:
    (tmp_path / "lib").mkdir()
    (tmp_path / "main.js").write_text(
        'import answer from "./lib/answer";\nexport function run() { return {output: answer}; }\n',
        encoding="utf-8",
    )
    (tmp_path / "lib" / "answer.mjs").write_text("export default 42;\n", encoding="utf-8")

    bundle = build_workflow_js_module_bundle(entry_module="main.js", local_roots=[tmp_path])

    assert bundle["ok"] is True
    assert bundle["resolved_modules"] == ["lib/answer.mjs", "main.js"]
    assert bundle["unresolved_imports"] == []


def test_workflow_js_module_bundle_resolves_parent_relative_imports() -> None:
    bundle = build_workflow_js_module_bundle(
        entry_module="features/main.js",
        modules=[
            {
                "id": "features/main.js",
                "source": 'import { answer } from "../shared/answer.js";\nexport function run() { return {output: answer}; }\n',
            },
            {"id": "shared/answer.js", "source": "export const answer = 42;\n"},
        ],
    )

    assert bundle["ok"] is True
    assert bundle["resolved_allowed_imports"] == ["../shared/answer.js"]


def test_workflow_js_module_bundle_reports_disabled_and_unresolved_libs(tmp_path) -> None:
    disabled_root = tmp_path / "disabled"
    disabled_root.mkdir()
    (disabled_root / "blocked.js").write_text("export const value = 1;\n", encoding="utf-8")
    bundle = build_workflow_js_module_bundle(
        entry_module="main.js",
        disabled_lib_roots=[disabled_root],
        modules=[
            {
                "id": "main.js",
                "source": (
                    'import { value } from "blocked";\n'
                    'import missing from "missing-lib";\n'
                    "export function run() { return {output: value || missing}; }\n"
                ),
            }
        ],
    )

    assert bundle["ok"] is False
    assert bundle["resolved_disabled_imports"] == ["blocked"]
    assert bundle["unresolved_imports"] == ["missing-lib"]


def test_workflow_js_module_bundle_resolves_allowed_lib_imports(tmp_path) -> None:
    lib_root = tmp_path / "libs"
    (lib_root / "math").mkdir(parents=True)
    (lib_root / "math" / "index.js").write_text("import { two } from './two.js';\nexport const answer = two * 21;\n", encoding="utf-8")
    (lib_root / "math" / "two.js").write_text("export const two = 2;\n", encoding="utf-8")
    bundle = build_workflow_js_module_bundle(
        entry_module="main.js",
        allowed_lib_roots=[lib_root],
        modules=[
            {
                "id": "main.js",
                "source": 'import { answer } from "math";\nexport function run() { return {output: answer}; }\n',
            }
        ],
    )

    assert bundle["ok"] is True
    assert bundle["resolved_modules"] == ["lib:0:math/index.js", "lib:0:math/two.js", "main.js"]

    out = WorkflowJsNodeRuntimeRegistry().execute(
        {
            "request_id": "req-module-bundle-lib-root",
            "module_source": bundle["module_source"],
            "module_sha256": bundle["module_sha256"],
            "package_id": "pkg",
            "workflow_id": "wf",
            "package_source_digest": "digest",
            "payload": {},
            "limits": {"timeout_ms": 5000, "output_limit_bytes": 65536, "memory_limit_mb": 128},
        }
    )

    assert out["ok"] is True
    assert out["output"] == 42


def test_workflow_js_module_bundle_resolves_allowed_lib_index_mjs_and_relative_mjs(tmp_path) -> None:
    lib_root = tmp_path / "libs"
    (lib_root / "math").mkdir(parents=True)
    (lib_root / "math" / "index.mjs").write_text("import { two } from './two';\nexport const answer = two * 21;\n", encoding="utf-8")
    (lib_root / "math" / "two.mjs").write_text("export const two = 2;\n", encoding="utf-8")
    bundle = build_workflow_js_module_bundle(
        entry_module="main.js",
        allowed_lib_roots=[lib_root],
        modules=[
            {
                "id": "main.js",
                "source": 'import { answer } from "math";\nexport function run() { return {output: answer}; }\n',
            }
        ],
    )

    assert bundle["ok"] is True
    assert bundle["resolved_modules"] == ["lib:0:math/index.mjs", "lib:0:math/two.mjs", "main.js"]
    assert bundle["unresolved_imports"] == []


def test_workflow_js_module_bundle_rejects_require_and_node_builtins() -> None:
    bundle = build_workflow_js_module_bundle(
        entry_module="main.js",
        modules=[
            {
                "id": "main.js",
                "source": 'import fs from "node:fs";\nconst x = require("x");\nexport function run() { return {output: x}; }\n',
            }
        ],
    )

    assert bundle["ok"] is False
    assert {"module": "main.js", "specifier": "require(...)", "reason": "require_unsupported"} in bundle["rejected_imports"]
    assert {"module": "main.js", "specifier": "node:fs", "reason": "node_builtin_unsupported"} in bundle["rejected_imports"]
