from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import shutil
import subprocess
import sys
import threading
import venv
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from functools import lru_cache
from pathlib import Path

import pytest

from hosting.service.toolbox_materialization import HermeticToolboxTemplateMaterializer
from hosting.service.host_service import EngineHostService
from hosting.toolbox.catalog import (
    ToolboxEnvironmentTemplateSpec,
    ToolboxLockedDistributionSpec,
    ToolboxTemplateProvenance,
)
from hosting.toolbox.bundle_models import (
    SandboxProfileSpec,
    ToolboxAutoAssignmentRequest,
    ToolboxBundleFile,
)
from hosting.toolbox.hermetic_environment import (
    HermeticToolboxEnvironmentBuildError,
    PythonEnvironmentBuilder,
    ResolvedToolboxEnvironmentInput,
    ToolboxLockedArtifactSpec,
    HermeticToolboxEnvironmentResolver,
)
from hosting.toolbox.orchestration import ToolboxSandboxOrchestrator
from hosting.toolbox.staging import ToolboxBundleStager
from hosting.toolbox.target import detect_current_toolbox_target


TARGET = detect_current_toolbox_target()


def _digest(character: str) -> str:
    return "sha256:" + character * 64


def _wheel(source: Path, distribution: str, version: str, import_root: str) -> ToolboxLockedArtifactSpec:
    filename = f"{distribution.replace('-', '_')}-{version}-py3-none-any.whl"
    target = source / filename
    raw = _wheel_bytes(distribution, version, import_root)
    target.write_bytes(raw)
    return ToolboxLockedArtifactSpec(
        distribution_name=distribution,
        version=version,
        source_id="approved",
        filename=filename,
        sha256=f"sha256:{hashlib.sha256(raw).hexdigest()}",
        size_bytes=len(raw),
    )


@lru_cache(maxsize=None)
def _wheel_bytes(distribution: str, version: str, import_root: str) -> bytes:
    import io

    dist_info = f"{distribution.replace('-', '_')}-{version}.dist-info"
    members = {
        f"{import_root}/__init__.py": f"VALUE = {version!r}\n",
        f"{dist_info}/METADATA": (
            "Metadata-Version: 2.1\n"
            f"Name: {distribution}\n"
            f"Version: {version}\n"
        ),
        f"{dist_info}/WHEEL": (
            "Wheel-Version: 1.0\n"
            "Generator: mp13-test\n"
            "Root-Is-Purelib: true\n"
            "Tag: py3-none-any\n"
        ),
    }
    members[f"{dist_info}/RECORD"] = "".join(f"{name},,\n" for name in members) + f"{dist_info}/RECORD,,\n"
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in members.items():
            archive.writestr(name, content)
    return output.getvalue()


@pytest.fixture(scope="session")
def hermetic_venv_seed(tmp_path_factory: pytest.TempPathFactory) -> Path:
    seed = tmp_path_factory.mktemp("hermetic-venv-seed") / "environment"
    venv.EnvBuilder(with_pip=True, system_site_packages=False, clear=True).create(str(seed))
    return seed


@pytest.fixture(autouse=True)
def reuse_immutable_hermetic_venv_seed(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
    hermetic_venv_seed: Path,
) -> None:
    if request.node.get_closest_marker("real_venv_creation") is not None:
        return

    def _copy_seed(_builder: venv.EnvBuilder, target: str) -> None:
        shutil.copytree(hermetic_venv_seed, target, dirs_exist_ok=True)

    monkeypatch.setattr(venv.EnvBuilder, "create", _copy_seed)


def _resolved(
    artifacts: tuple[ToolboxLockedArtifactSpec, ...],
    *,
    template_id: str = "core",
    imports: tuple[str, ...] = ("alpha_pkg",),
    custom_digest: str | None = None,
    lock_digest: str = _digest("3"),
) -> ResolvedToolboxEnvironmentInput:
    ordered = tuple(sorted(artifacts))
    return ResolvedToolboxEnvironmentInput(
        template_id=template_id,
        template_digest=_digest("1"),
        runtime_version=".".join(str(item) for item in sys.version_info[:3]),
        runtime_artifact_digest=_digest("2"),
        python_abi=TARGET.python_abi,
        platform=TARGET.platform,
        complete_lock_digest=lock_digest,
        complete_lock=tuple(
            ToolboxLockedDistributionSpec(name=item.distribution_name, version=item.version)
            for item in ordered
        ),
        locked_artifacts=ordered,
        custom_resolved_lock_digest=custom_digest,
        isolation_policy_version="toolbox-isolation-v1",
        resolved_import_roots=imports,
    )


def _run(python: str, source: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment["PYTHONNOUSERSITE"] = "1"
    return subprocess.run(
        [python, "-c", source], check=False, capture_output=True, text=True, env=environment
    )


def _materialize_in_process(host: str, source: str, payload: dict, queue) -> None:
    try:
        builder = PythonEnvironmentBuilder(
            Path(host), artifact_sources={"approved": Path(source)}
        )
        spec = builder.materialize_environment(payload)
        queue.put({"ok": True, "environment_root": spec.environment_root})
    except Exception as exc:  # pragma: no cover - asserted through child result
        queue.put({"ok": False, "error": type(exc).__name__, "detail": str(exc)})


def test_builder_rejects_cross_target_wheel_before_source_access(tmp_path: Path) -> None:
    source = tmp_path / "approved"
    source.mkdir()
    portable = _wheel(source, "alpha-package", "1.0.0", "alpha_pkg")
    foreign_platform = "win_arm64" if TARGET.platform != "win_arm64" else "win_amd64"
    foreign = replace(
        portable,
        filename=f"alpha_package-1.0.0-cp312-cp312-{foreign_platform}.whl",
    )
    resolved = _resolved((foreign,))
    builder = PythonEnvironmentBuilder(
        tmp_path / "host", artifact_sources={"approved": tmp_path / "missing-source"}
    )

    with pytest.raises(HermeticToolboxEnvironmentBuildError) as captured:
        builder.materialize_environment(resolved)

    assert captured.value.code == "environment_artifact_target_mismatch"


@pytest.mark.real_venv_creation
def test_offline_preseed_builds_non_inheriting_venv_and_publishes_verified_receipt(tmp_path: Path) -> None:
    source = tmp_path / "approved"
    source.mkdir()
    alpha = _wheel(source, "alpha-package", "1.0.0", "alpha_pkg")
    resolved = _resolved((alpha,))
    builder = PythonEnvironmentBuilder(tmp_path / "host", artifact_sources={"approved": source})

    spec = builder.materialize_environment(resolved)

    assert Path(spec.python_executable).is_file()
    config = (Path(spec.environment_root) / "pyvenv.cfg").read_text(encoding="utf-8").lower()
    assert "include-system-site-packages = false" in config
    imported = _run(spec.python_executable, "import alpha_pkg; print(alpha_pkg.VALUE)")
    assert imported.returncode == 0
    assert imported.stdout.strip() == "1.0.0"
    ambient = _run(spec.python_executable, "import importlib.util; raise SystemExit(0 if importlib.util.find_spec('pytest') is None else 9)")
    assert ambient.returncode == 0
    receipt = json.loads((Path(spec.environment_root) / "verification-receipt.json").read_text(encoding="utf-8"))
    assert receipt["state"] == "verified"
    assert receipt["system_site_packages"] is False
    assert receipt["installed_distributions"] == {"alpha-package": "1.0.0"}
    assert receipt["verified_import_roots"] == ["alpha_pkg"]


def test_failed_final_interpreter_probe_is_quarantined_and_never_published(tmp_path: Path) -> None:
    source = tmp_path / "approved"
    source.mkdir()
    alpha = _wheel(source, "alpha-package", "1.0.0", "alpha_pkg")
    resolved = _resolved((alpha,), imports=("missing_root",))
    builder = PythonEnvironmentBuilder(tmp_path / "host", artifact_sources={"approved": source})

    with pytest.raises(HermeticToolboxEnvironmentBuildError, match="environment_import_probe_failed"):
        builder.materialize_environment(resolved)

    spec = builder.resolver.environment_spec(resolved)
    assert not Path(spec.environment_root).exists()
    failures = list(builder.quarantine_root.glob("*/failure.json"))
    assert len(failures) == 1
    assert json.loads(failures[0].read_text(encoding="utf-8"))["code"] == "environment_import_probe_failed"


def test_artifact_digest_mismatch_fails_before_install(tmp_path: Path) -> None:
    source = tmp_path / "approved"
    source.mkdir()
    alpha = _wheel(source, "alpha-package", "1.0.0", "alpha_pkg")
    bad = ToolboxLockedArtifactSpec(**{**alpha.to_dict(), "sha256": _digest("f")})
    builder = PythonEnvironmentBuilder(tmp_path / "host", artifact_sources={"approved": source})

    with pytest.raises(HermeticToolboxEnvironmentBuildError, match="environment_artifact_verification_failed"):
        builder.materialize_environment(_resolved((bad,)))


def test_concurrent_requests_deduplicate_one_physical_build(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "approved"
    source.mkdir()
    alpha = _wheel(source, "alpha-package", "1.0.0", "alpha_pkg")
    resolved = _resolved((alpha,))
    builder = PythonEnvironmentBuilder(tmp_path / "host", artifact_sources={"approved": source})
    original = builder._build_candidate
    count = 0
    count_lock = threading.Lock()

    def counted(*args, **kwargs):
        nonlocal count
        with count_lock:
            count += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(builder, "_build_candidate", counted)
    with ThreadPoolExecutor(max_workers=4) as pool:
        specs = list(pool.map(lambda _index: builder.materialize_environment(resolved), range(4)))

    assert count == 1
    assert len({item.environment_root for item in specs}) == 1


def test_concurrent_processes_share_one_atomically_published_environment(tmp_path: Path) -> None:
    source = tmp_path / "approved"
    source.mkdir()
    alpha = _wheel(source, "alpha-package", "1.0.0", "alpha_pkg")
    resolved = _resolved((alpha,))
    host = tmp_path / "host"
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    processes = [
        context.Process(
            target=_materialize_in_process,
            args=(str(host), str(source), resolved.to_dict(), queue),
        )
        for index in range(2)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(120)
        assert process.exitcode == 0
    results = [queue.get(timeout=5) for _ in processes]

    assert all(item["ok"] for item in results), results
    assert len({item["environment_root"] for item in results}) == 1
    builder = PythonEnvironmentBuilder(host, artifact_sources={"approved": source})
    assert builder.verified_environment(resolved).environment_root == results[0]["environment_root"]
    assert not list(builder.candidates_root.glob("*"))


def test_complete_base_plus_delta_lock_is_independent_of_base_site_packages(tmp_path: Path) -> None:
    source = tmp_path / "approved"
    source.mkdir()
    alpha = _wheel(source, "alpha-package", "1.0.0", "alpha_pkg")
    delta = _wheel(source, "delta-package", "2.0.0", "delta_pkg")
    builder = PythonEnvironmentBuilder(tmp_path / "host", artifact_sources={"approved": source})
    base = _resolved((alpha,), lock_digest=_digest("3"))
    derived = _resolved(
        (alpha, delta),
        template_id="custom",
        imports=("alpha_pkg", "delta_pkg"),
        custom_digest=_digest("4"),
        lock_digest=_digest("5"),
    )

    base_spec = builder.materialize_environment(base)
    derived_spec = builder.materialize_environment(derived)

    assert base_spec.environment_root != derived_spec.environment_root
    check = _run(
        derived_spec.python_executable,
        f"import alpha_pkg,delta_pkg,sys; raise SystemExit(8 if {str(base_spec.environment_root)!r} in sys.path else 0)",
    )
    assert check.returncode == 0
    derived_receipt = json.loads((Path(derived_spec.environment_root) / "verification-receipt.json").read_text(encoding="utf-8"))
    assert derived_receipt["installed_distributions"] == {
        "alpha-package": "1.0.0",
        "delta-package": "2.0.0",
    }


def test_required_environment_readiness_is_receipt_gated(tmp_path: Path) -> None:
    source = tmp_path / "approved"
    source.mkdir()
    alpha = _wheel(source, "alpha-package", "1.0.0", "alpha_pkg")
    resolved = _resolved((alpha,))
    builder = PythonEnvironmentBuilder(tmp_path / "host", artifact_sources={"approved": source})

    assert builder.required_environment_readiness((resolved,))["state"] == "degraded"
    builder.materialize_environment(resolved)
    readiness = builder.required_environment_readiness((resolved,))
    assert readiness["state"] == "ready"
    assert readiness["environments"][0]["code"] == "environment_verified"


def test_catalog_prewarm_adapter_builds_complete_wheel_lock_on_target_host(tmp_path: Path) -> None:
    source = tmp_path / "approved"
    source.mkdir()
    alpha = _wheel(source, "alpha-package", "1.0.0", "alpha_pkg")
    template = ToolboxEnvironmentTemplateSpec(
        template_id="core",
        python_requires=">=3.12,<3.13",
        python_abis=(TARGET.python_abi,),
        runtime_kind="toolbox_python",
        worker_protocol_version="1.0.0",
        platforms=(TARGET.platform,),
        locked_distributions=(ToolboxLockedDistributionSpec(name="alpha-package", version="1.0.0"),),
        exposed_import_roots=("alpha_pkg",),
        lock_digest=_digest("3"),
        parent_worker_artifact_digest=_digest("2"),
        isolation_policy_version="1.0.0",
        provenance=ToolboxTemplateProvenance(
            source="test",
            revision="one",
            evidence_digest=_digest("7"),
            verifier_id="test-verifier",
        ),
    )
    entry = {
        "template_id": "core",
        "template_digest": _digest("1"),
        "template": template.to_dict(),
        "artifacts": [
            {
                "source_id": alpha.source_id,
                "filename": alpha.filename,
                "sha256": alpha.sha256,
                "size_bytes": alpha.size_bytes,
            }
        ],
    }
    builder = PythonEnvironmentBuilder(tmp_path / "host", artifact_sources={"approved": source})
    materializer = HermeticToolboxTemplateMaterializer(builder)
    progress: list[tuple] = []

    receipt = materializer.materialize(
        catalog_entry=entry,
        python_abi=template.python_abis[0],
        platform=template.platforms[0],
        progress=lambda *items: progress.append(items),
    )

    assert receipt.template_id == "core"
    assert receipt.artifact_digests == (alpha.sha256,)
    assert receipt.verified_import_roots == ("alpha_pkg",)
    assert [item[1] for item in progress] == ["hermetic_environment_building", "hermetic_environment_verified"]


def test_service_rejects_legacy_artifact_source_constructor_inputs(tmp_path: Path) -> None:
    source = tmp_path / "approved"
    source.mkdir()
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        EngineHostService(  # type: ignore[call-arg]
            toolbox_artifact_sources={"approved": source},
        )


def test_configured_orchestrator_spawns_only_with_resolved_verified_interpreter(tmp_path: Path) -> None:
    source = tmp_path / "approved"
    source.mkdir()
    alpha = _wheel(source, "alpha-package", "1.0.0", "alpha_pkg")
    resolved = _resolved((alpha,))
    hermetic = HermeticToolboxEnvironmentResolver(tmp_path / "host").environment_spec(resolved)

    class Service:
        _hermetic_toolbox_environment_builder = object()
        _toolbox_required_python_abi = resolved.python_abi
        _toolbox_required_platform = resolved.platform

        def __init__(self):
            self.materialized: list[dict] = []
            self.spawned: list[dict] = []

        def materialize_toolbox_environment_for_bundle(self, **kwargs):
            self.materialized.append(kwargs)
            return hermetic

        def spawn(self, **kwargs):
            self.spawned.append(kwargs)
            return {"engine_id": kwargs["engine_id"], "environment": kwargs["environment"]}

    service = Service()
    orchestrator = ToolboxSandboxOrchestrator(
        service=service,
        stager=ToolboxBundleStager(tmp_path / "host"),
        python_executable="C:/forbidden-bootstrap/python.exe",
    )
    assignments = orchestrator.spawn_assignments(
        toolbox_id="strict",
        requests=[
            ToolboxAutoAssignmentRequest(
                files=[ToolboxBundleFile(relative_path="alpha.py", content="def alpha():\n    return 1\n")],
                module_name="alpha",
                callable_name="alpha",
                sandbox_profile=SandboxProfileSpec(required_imports=["alpha_pkg"]),
            )
        ],
    )

    assert len(assignments) == 1
    assert len(service.materialized) == 1
    assert service.spawned[0]["command"][0] == hermetic.python_executable
    assert service.spawned[0]["command"][0] != "C:/forbidden-bootstrap/python.exe"
    assert service.spawned[0]["environment"]["venv_key"] == resolved.environment_key
