from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

from hosting.toolbox.builtin_resolver import AirgapBuiltinWheelResolver
from hosting.toolbox.host_project_config import ToolboxHostProjectConfiguration


def _wheel(
    root: Path,
    name: str,
    version: str,
    *,
    requires: tuple[str, ...] = (),
    tag: str = "py3-none-any",
) -> Path:
    distribution = name.replace("-", "_")
    filename = f"{distribution}-{version}-{tag}.whl"
    dist_info = f"{distribution}-{version}.dist-info"
    metadata = ["Metadata-Version: 2.1", f"Name: {name}", f"Version: {version}"]
    metadata.extend(f"Requires-Dist: {item}" for item in requires)
    path = root / filename
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(f"{distribution}/__init__.py", "")
        archive.writestr(f"{dist_info}/METADATA", "\n".join(metadata) + "\n")
        archive.writestr(
            f"{dist_info}/WHEEL",
            "Wheel-Version: 1.0\nGenerator: mp13-test\nRoot-Is-Purelib: true\n"
            f"Tag: {tag}\n",
        )
        archive.writestr(f"{dist_info}/RECORD", "")
    return path


def _configuration(*, requirements: tuple[str, ...], source_ids: tuple[str, ...] = ("primary",)) -> dict:
    return {
        "builtins": [
            {
                "template_id": "core",
                "imports": ["alpha"],
                "package_requirements": list(requirements),
                "sandbox_policy": "compute-only",
                "required": True,
                "prewarm": True,
                "provenance": "test-release",
            }
        ],
        "sources": [
            {
                "source_id": source_id,
                "kind": "airgap_store",
                "origin": f"airgap://{source_id}",
                "credential_ref": None,
                "allowed_package_namespaces": ["*"],
                "priority": 100 - index,
                "trust_key_ids": ["test-key"],
                "maximum_download_bytes": 10_000_000,
            }
            for index, source_id in enumerate(source_ids)
        ],
        "resolution": {
            "mode": "air_gapped",
            "timeout_seconds": 30,
            "maximum_bytes": 10_000_000,
            "maximum_artifacts": 10,
            "allowed_redirect_origins": [],
            "wheel_only": True,
        },
        "retention": {
            "artifact_cache_grace_seconds": 60,
            "maximum_cache_bytes": 10_000_000,
            "maximum_cache_artifacts": 100,
            "protected_digests": [],
            "remove_unreferenced_custom_revisions_on_apply": False,
        },
    }


def test_airgap_resolver_returns_exact_transitive_current_host_closure(tmp_path: Path) -> None:
    wheelhouse = tmp_path / "primary"
    wheelhouse.mkdir()
    alpha = _wheel(wheelhouse, "alpha", "1.0", requires=("beta==2.0",))
    beta = _wheel(wheelhouse, "beta", "2.0")
    configuration = ToolboxHostProjectConfiguration.from_dict(
        _configuration(requirements=("alpha==1.0",))
    )

    result = AirgapBuiltinWheelResolver(
        configuration, artifact_sources={"primary": wheelhouse}
    ).resolve()

    assert result.status == "resolved"
    assert result.diagnostics == ()
    closure = result.closures[0]
    assert [(item.name, item.version) for item in closure.locked_distributions] == [
        ("alpha", "1.0"),
        ("beta", "2.0"),
    ]
    assert [item.filename for item in closure.locked_artifacts] == [alpha.name, beta.name]
    assert [item.sha256 for item in closure.locked_artifacts] == [
        "sha256:" + hashlib.sha256(alpha.read_bytes()).hexdigest(),
        "sha256:" + hashlib.sha256(beta.read_bytes()).hexdigest(),
    ]
    assert AirgapBuiltinWheelResolver(
        configuration, artifact_sources={"primary": wheelhouse}
    ).resolve().to_dict() == result.to_dict()


def test_required_missing_wheel_returns_stable_result_and_no_partial_closure(tmp_path: Path) -> None:
    wheelhouse = tmp_path / "primary"
    wheelhouse.mkdir()
    _wheel(wheelhouse, "alpha", "1.0", requires=("missing-child==9.0",))
    configuration = ToolboxHostProjectConfiguration.from_dict(
        _configuration(requirements=("alpha==1.0",))
    )

    result = AirgapBuiltinWheelResolver(
        configuration, artifact_sources={"primary": wheelhouse}
    ).resolve()

    assert result.status == "not_ready"
    assert result.closures == ()
    assert result.diagnostics == (
        {
            "template_id": "core",
            "code": "required_template_wheel_missing",
            "summary": "No complete compatible exact wheel closure is available.",
        },
    )


def test_incompatible_wheel_is_reported_as_missing_without_path_leak(tmp_path: Path) -> None:
    wheelhouse = tmp_path / "primary"
    wheelhouse.mkdir()
    _wheel(wheelhouse, "alpha", "1.0", tag="cp311-cp311-win_amd64")
    configuration = ToolboxHostProjectConfiguration.from_dict(
        _configuration(requirements=("alpha==1.0",))
    )

    result = AirgapBuiltinWheelResolver(
        configuration, artifact_sources={"primary": wheelhouse}
    ).resolve()

    assert result.status == "not_ready"
    assert result.diagnostics[0]["code"] == "required_template_wheel_missing"
    assert str(tmp_path) not in str(result.to_dict())


def test_any_required_failure_discards_other_resolved_intents(tmp_path: Path) -> None:
    wheelhouse = tmp_path / "primary"
    wheelhouse.mkdir()
    _wheel(wheelhouse, "alpha", "1.0")
    payload = _configuration(requirements=("alpha==1.0",))
    payload["builtins"].append(
        {
            **payload["builtins"][0],
            "template_id": "py-compute",
            "package_requirements": ["absent==1.0"],
        }
    )
    configuration = ToolboxHostProjectConfiguration.from_dict(payload)

    result = AirgapBuiltinWheelResolver(
        configuration, artifact_sources={"primary": wheelhouse}
    ).resolve()

    assert result.status == "not_ready"
    assert result.closures == ()
    assert [item["template_id"] for item in result.diagnostics] == ["py-compute"]
