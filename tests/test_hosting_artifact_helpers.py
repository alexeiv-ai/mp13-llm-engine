from __future__ import annotations

import base64
import io
import zipfile
from pathlib import Path

from hosting.sandbox.artifacts import (
    HostedArtifactManager,
    artifact_file_output,
    artifact_host_takeover_output,
    artifact_inline_input,
    artifact_inline_zip_input,
    artifact_inline_zip_output,
    artifact_masked_ref_input,
    artifact_producer_owned_output,
    artifact_ref_input,
)


def test_artifact_helper_rows_serialize_stable_defaults() -> None:
    inline = artifact_inline_input(name="seed", text="hello")
    inline_zip = artifact_inline_zip_input(name="project", base64_data="UEsDBAoAAAAAA")
    ref_input = artifact_ref_input(name="seed", ref="@project/seed.txt")
    masked = artifact_masked_ref_input(name="dataset", ref="@project/data", path_mask="*.txt")
    file_output = artifact_file_output(name="report", media_type="text/plain")
    takeover = artifact_host_takeover_output(name="report", ref="@project/out/report.txt")
    producer = artifact_producer_owned_output(name="report", ref="@project/out/report.txt")
    inline_zip_output = artifact_inline_zip_output(name="bundle", ref="@project/out", path_mask="*.py")

    assert inline == {
        "name": "seed",
        "kind": "inline",
        "filename": "seed.txt",
        "media_type": "text/plain",
        "encoding": "utf-8",
        "text": "hello",
    }
    assert inline_zip["filename"] == "project.zip"
    assert inline_zip["media_type"] == "application/zip"
    assert inline_zip["encoding"] == "zip"
    assert ref_input == {
        "name": "seed",
        "kind": "ref",
        "ref": "@project/seed.txt",
        "media_type": "application/octet-stream",
    }
    assert masked["recursive"] is True
    assert masked["path_mask"] == "*.txt"
    assert file_output["filename"] == "report.bin"
    assert takeover["host_takeover"] is True
    assert producer["ownership"] == "producer"
    assert inline_zip_output["export_inline_zip"] is True
    assert inline_zip_output["filename"] == "bundle.zip"


def test_artifact_templates_prepare_collect_and_cleanup_ownership(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    (project_root / "seed.txt").write_text("seed", encoding="utf-8")
    artifact_root = tmp_path / "artifacts"
    manager = HostedArtifactManager(artifact_root=artifact_root, artifact_roots={"project": project_root})
    request = {
        "artifact_inputs": [artifact_ref_input(name="seed", ref="@project/seed.txt")],
        "artifact_outputs": [
            artifact_file_output(name="hosted", filename="hosted.txt", media_type="text/plain"),
            artifact_producer_owned_output(name="producer", ref="@project/producer.txt", media_type="text/plain"),
            artifact_host_takeover_output(name="takeover", ref="@project/takeover.txt", media_type="text/plain"),
        ],
    }

    context = manager.prepare(request=request, request_id="req-artifact-helpers")
    child = context["child_context"]
    Path(child["outputs"]["hosted"]).write_text("hosted", encoding="utf-8")
    Path(child["outputs"]["producer"]).write_text("producer", encoding="utf-8")
    Path(child["outputs"]["takeover"]).write_text("takeover", encoding="utf-8")

    assert Path(child["inputs"]["seed"]).read_text(encoding="utf-8") == "seed"
    artifacts = {row["name"]: row for row in manager.collect(context, request_id="req-artifact-helpers")}
    cleanup = manager.cleanup_run(context)

    assert artifacts["hosted"]["ref"].startswith("@artifacts/")
    assert artifacts["producer"]["ref"] == "@project/producer.txt"
    assert artifacts["takeover"]["ref"].startswith("@artifacts/")
    assert artifacts["takeover"]["ownership"] == "host"
    assert (project_root / "producer.txt").read_text(encoding="utf-8") == "producer"
    assert not (project_root / "takeover.txt").exists()
    assert cleanup["status"] == "ok"
    assert cleanup["deleted"] is True
    assert not Path(context["run_root"]).exists()


def test_artifact_inline_zip_templates_expand_and_export(tmp_path: Path) -> None:
    raw = io.BytesIO()
    with zipfile.ZipFile(raw, "w") as zf:
        zf.writestr("pkg/a.py", "A = 1")
        zf.writestr("pkg/b.py", "B = 2")
    manager = HostedArtifactManager(artifact_root=tmp_path / "artifacts")
    request = {
        "artifact_inputs": [
            artifact_inline_zip_input(
                name="project",
                base64_data=base64.b64encode(raw.getvalue()).decode("ascii"),
            )
        ],
        "artifact_outputs": [artifact_inline_zip_output(name="bundle", path_mask="*.py")],
    }
    context = manager.prepare(request=request, request_id="req-artifact-zip")
    output_root = Path(context["child_context"]["outputs"]["bundle"])
    (output_root / "pkg").mkdir(parents=True)
    (output_root / "pkg" / "c.py").write_text("C = 3", encoding="utf-8")

    artifacts = manager.collect(context, request_id="req-artifact-zip")

    assert sorted(path.name for path in Path(context["child_context"]["inputs"]["project"]).glob("pkg/*.py")) == ["a.py", "b.py"]
    assert len(artifacts) == 1
    assert artifacts[0]["kind"] == "inline"
    assert artifacts[0]["media_type"] == "application/zip"
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(artifacts[0]["base64"])), "r") as zf:
        assert zf.namelist() == ["pkg/c.py"]
