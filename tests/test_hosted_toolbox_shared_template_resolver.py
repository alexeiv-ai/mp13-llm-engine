from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import pytest

from hosting.service.host_service import EngineHostService
from hosting.service.toolbox_materialization import (
    ToolboxTemplateMaterializationReceipt,
    derived_environment_digest,
)


ROOT = Path(__file__).resolve().parents[1]


class ReceiptMaterializer:
    def materialize(self, *, catalog_entry: Mapping[str, Any], python_abi: str, platform: str, progress):
        artifacts = tuple(sorted(item["sha256"] for item in catalog_entry["artifacts"]))
        roots = tuple(sorted(catalog_entry["template"]["exposed_import_roots"]))
        progress("artifact_verification", "resolver_artifacts_verified", 1, 1, "Verified artifacts.", True)
        progress("environment_build", "resolver_environment_built", 1, 1, "Built environment.", True)
        progress("import_probe", "resolver_imports_probed", len(roots), len(roots), "Probed imports.", False)
        return ToolboxTemplateMaterializationReceipt(
            template_id=catalog_entry["template_id"],
            template_digest=catalog_entry["template_digest"],
            python_abi=python_abi,
            platform=platform,
            environment_digest=derived_environment_digest(
                template_digest=catalog_entry["template_digest"],
                python_abi=python_abi,
                platform=platform,
                artifact_digests=artifacts,
            ),
            artifact_digests=artifacts,
            verified_import_roots=roots,
            verified_at_ms=int(time.time() * 1000),
            verifier="shared-resolver-test-v1",
        )


def _ready_service(tmp_path: Path) -> EngineHostService:
    service = EngineHostService(
        engines_state_file=tmp_path / "engines.json",
        control_state_file=tmp_path / "access_control.json",
        toolbox_template_materializer=ReceiptMaterializer(),
    )
    started = service.initialize_shipped_toolbox_templates(
        python_abi="cp312",
        platform="win_amd64",
        request_id_prefix="resolver-setup",
    )
    for operation in started["operations"]:
        terminal = service._hosted_operations.wait_for_terminal(  # noqa: SLF001
            operation_id=operation["operation"]["operation_id"], timeout_seconds=10
        )
        assert terminal["lifecycle"] == "terminal_success"
    return service


@pytest.mark.parametrize(
    "consumer_kind",
    [
        "toolbox",
        "workflow_python_node",
        "workflow_python_snippet",
        "workflow_python_helper",
    ],
)
def test_all_python_consumers_resolve_core_from_same_verified_receipt(
    tmp_path: Path, consumer_kind: str
) -> None:
    service = _ready_service(tmp_path)
    resolution = service.resolve_hosted_template_environment(
        consumer_kind=consumer_kind,
        files=[{"relative_path": "main.py", "content": "import json\nVALUE = 1\n"}],
        python_abi="cp312",
        platform="win_amd64",
    )
    binding = resolution["binding"]
    assert binding["consumer_kind"] == consumer_kind
    assert binding["template_id"] == "core"
    assert binding["environment_digest"].startswith("sha256:")
    assert resolution["requirements"] == []


def test_consumer_bindings_share_physical_receipt_without_aliasing_runtime_identity(tmp_path: Path) -> None:
    service = _ready_service(tmp_path)
    bindings = {}
    for consumer in ["toolbox", "workflow_python_node", "workflow_python_snippet", "workflow_python_helper"]:
        bindings[consumer] = service.resolve_hosted_template_environment(
            consumer_kind=consumer,
            files=[{"relative_path": "main.py", "content": "import json\n"}],
            python_abi="cp312",
            platform="win_amd64",
        )["binding"]
    assert len({item["environment_digest"] for item in bindings.values()}) == 1
    assert len({item["template_digest"] for item in bindings.values()}) == 1
    assert len({item["binding_id"] for item in bindings.values()}) == 4
    assert bindings["workflow_python_node"]["runtime_family"] == "workflow_python_node"
    assert bindings["workflow_python_snippet"]["runtime_family"] == "workflow_python_node"
    assert bindings["workflow_python_node"]["binding_id"] != bindings["workflow_python_snippet"]["binding_id"]
    assert bindings["workflow_python_helper"]["runtime_family"] == "workflow_python_helper"
    assert bindings["toolbox"]["runtime_family"] == "toolbox_executor"


def test_intrinsic_resolution_uses_py_compute_and_is_read_only(tmp_path: Path) -> None:
    service = _ready_service(tmp_path)
    operations_before = (tmp_path / "state" / "hosted_operations.json").read_bytes()
    engines_before = service._read_engines()  # noqa: SLF001
    resolution = service.resolve_hosted_template_environment(
        consumer_kind="toolbox",
        files=[{"relative_path": "main.py", "content": "VALUE = 1\n"}],
        intrinsic_names=["scriptable_calculator", "symbolic_algebra"],
        python_abi="cp312",
        platform="win_amd64",
    )
    assert resolution["binding"]["template_id"] == "py-compute"
    assert {item["distribution"] for item in resolution["requirements"]} == {
        "numexpr", "numpy", "sympy"
    }
    assert service._read_engines() == engines_before  # noqa: SLF001
    assert (tmp_path / "state" / "hosted_operations.json").read_bytes() == operations_before


def test_unverified_templates_are_not_resolvable(tmp_path: Path) -> None:
    service = EngineHostService(
        engines_state_file=tmp_path / "engines.json",
        control_state_file=tmp_path / "access_control.json",
    )
    with pytest.raises(ValueError, match="verified_template_target_unavailable"):
        service.resolve_hosted_template_environment(
            consumer_kind="workflow_python_helper",
            files=[{"relative_path": "main.py", "content": "import json\n"}],
            python_abi="cp312",
            platform="win_amd64",
        )


def test_sandbox_policy_changes_binding_not_physical_environment(tmp_path: Path) -> None:
    service = _ready_service(tmp_path)
    common = {
        "consumer_kind": "workflow_python_node",
        "files": [{"relative_path": "main.py", "content": "import json\n"}],
        "python_abi": "cp312",
        "platform": "win_amd64",
    }
    default = service.resolve_hosted_template_environment(**common)["binding"]
    narrowed = service.resolve_hosted_template_environment(
        **common,
        sandbox_policy={"policy_id": "compute-only-narrowed", "network": False},
    )["binding"]
    assert default["environment_digest"] == narrowed["environment_digest"]
    assert default["binding_id"] != narrowed["binding_id"]
    assert default["sandbox_policy_digest"] != narrowed["sandbox_policy_digest"]


def test_isolated_process_probes_core_and_every_shipped_compute_intrinsic(tmp_path: Path) -> None:
    core = subprocess.run(
        [sys.executable, "-I", "-c", "import json,math,pathlib; print(json.dumps({'ok': math.sqrt(81)}))"],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
        env={**os.environ, "PYTHONNOUSERSITE": "1"},
    )
    assert core.returncode == 0, core.stderr
    assert json.loads(core.stdout)["ok"] == 9

    project_python = ROOT / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    executable = project_python if project_python.exists() else Path(sys.executable)
    code = (
        "import sys;"
        f"sys.path.insert(0,{str(ROOT / 'src')!r});"
        "from mp13_engine.mp13_tools_builtin import "
        "scriptable_calculator,scriptable_calculator_guide,symbolic_algebra,symbolic_algebra_guide;"
        "assert symbolic_algebra('(x+1)**2',['x'],'expand') == 'x**2 + 2*x + 1';"
        "assert scriptable_calculator('2 + 3')['result'] == 5;"
        "assert scriptable_calculator_guide('help');"
        "assert symbolic_algebra_guide('help');"
        "print('all-intrinsics-ok')"
    )
    compute = subprocess.run(
        [str(executable), "-I", "-c", code],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        timeout=60,
        check=False,
        env={**os.environ, "PYTHONNOUSERSITE": "1"},
    )
    assert compute.returncode == 0, compute.stderr
    assert compute.stdout.strip() == "all-intrinsics-ok"
