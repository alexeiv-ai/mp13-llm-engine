from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from hosting.toolbox.identity import (
    bundle_manifest_digest,
    canonical_json_bytes,
    custom_lock_digest,
    definition_revision,
    environment_identity,
    resolved_profile_identity,
    template_lock_digest,
)


ROOT = Path(__file__).resolve().parents[1]
VECTORS_PATH = ROOT / "src" / "hosting" / "toolbox" / "HOSTED_TOOLBOX_HASH_VECTORS.json"


def _vectors() -> list[dict]:
    payload = json.loads(VECTORS_PATH.read_text(encoding="utf-8"))
    assert payload["contract"] == "hosting.toolbox.identity_vectors.v1"
    return list(payload["vectors"])


def _calculate(name: str, value: dict) -> str:
    if name == "definition_revision":
        return definition_revision(value)
    if name == "resolved_profile_identity":
        return resolved_profile_identity(**value)
    if name == "environment_identity":
        return environment_identity(**value)
    if name == "bundle_manifest_digest":
        return bundle_manifest_digest(value)
    if name == "template_lock_digest":
        return template_lock_digest(value)
    if name == "custom_lock_digest":
        return custom_lock_digest(value)
    raise AssertionError(name)


def test_published_identity_vectors() -> None:
    for vector in _vectors():
        assert _calculate(vector["name"], vector["input"]) == vector["expected"]


def test_definition_revision_ignores_compare_revision_and_semantic_order() -> None:
    definition = copy.deepcopy(_vectors()[0]["input"])
    second = copy.deepcopy(definition["auto_requests"][0])
    second["module_name"] = "pkg.alpha"
    second["callable_name"] = "Alpha"
    second["files"] = [
        {"relative_path": "pkg/z.py", "content": "Z = 1\n"},
        {"relative_path": "pkg/alpha.py", "content": "def Alpha():\n    return 1\n"},
    ]
    definition["auto_requests"].append(second)
    expected = definition_revision(definition)
    definition["expected_revision"] = None
    definition["auto_requests"].reverse()
    definition["auto_requests"][0]["files"].reverse()
    definition["intrinsics"]["names"].reverse()
    for request in definition["auto_requests"]:
        request["dependency"]["declared_imports"].reverse()
    definition["toolbox_id"] = "café-tools"
    assert definition_revision(definition) == expected


def test_lock_and_manifest_record_order_is_not_semantic() -> None:
    by_name = {vector["name"]: copy.deepcopy(vector["input"]) for vector in _vectors()}
    template = by_name["template_lock_digest"]
    expected_template = template_lock_digest(template)
    template["distributions"].reverse()
    template["artifacts"].reverse()
    template["import_roots"].reverse()
    assert template_lock_digest(template) == expected_template

    manifest = by_name["bundle_manifest_digest"]
    expected_manifest = bundle_manifest_digest(manifest)
    manifest["files"].reverse()
    manifest["auto_tools"].reverse()
    manifest["manifest_hash"] = "sha256:" + "0" * 64
    manifest["bundle_revision"] = "0" * 16
    assert bundle_manifest_digest(manifest) == expected_manifest


def test_canonical_json_rejects_non_json_and_non_finite_values() -> None:
    with pytest.raises(ValueError, match="identity_number_must_be_finite"):
        canonical_json_bytes({"value": float("nan")})
    with pytest.raises(ValueError, match="identity_value_not_json"):
        canonical_json_bytes({"value": object()})
    with pytest.raises(ValueError, match="identity_object_key_must_be_string"):
        canonical_json_bytes({1: "value"})


def test_identity_vectors_are_stable_across_fresh_processes_and_hash_seeds() -> None:
    script = r"""
import json
import sys
from pathlib import Path
from hosting.toolbox.identity import (
    bundle_manifest_digest,
    custom_lock_digest,
    definition_revision,
    environment_identity,
    resolved_profile_identity,
    template_lock_digest,
)
functions = {
    "definition_revision": lambda value: definition_revision(value),
    "resolved_profile_identity": lambda value: resolved_profile_identity(**value),
    "environment_identity": lambda value: environment_identity(**value),
    "bundle_manifest_digest": lambda value: bundle_manifest_digest(value),
    "template_lock_digest": lambda value: template_lock_digest(value),
    "custom_lock_digest": lambda value: custom_lock_digest(value),
}
payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(json.dumps({row["name"]: functions[row["name"]](row["input"]) for row in payload["vectors"]}, sort_keys=True))
"""
    expected = {row["name"]: row["expected"] for row in _vectors()}
    outputs = []
    for seed in ("1", "987654"):
        env = dict(os.environ)
        env["PYTHONHASHSEED"] = seed
        env["PYTHONPATH"] = str(ROOT / "src")
        completed = subprocess.run(
            [sys.executable, "-c", script, str(VECTORS_PATH)],
            cwd=ROOT,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
        outputs.append(json.loads(completed.stdout))
    assert outputs == [expected, expected]
