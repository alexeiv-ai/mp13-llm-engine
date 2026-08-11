from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "hosting-native-targets.yml"


def test_native_target_workflow_covers_every_declared_runner_family() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    expected = {
        "windows-2025": "cp312-win_amd64",
        "windows-11-arm": "cp312-win_arm64",
        "ubuntu-24.04": "cp312-manylinux_2_28_x86_64",
        "ubuntu-24.04-arm": "cp312-manylinux_2_28_aarch64",
        "macos-15": "cp312-macosx_11_0_arm64",
    }
    for runner, target in expected.items():
        assert f"runner: {runner}" in text
        assert f"target: {target}" in text
    assert "pydantic-core==2.41.5" in text
    assert "cryptography==49.0.0" in text
    assert "tests/native_hosting_target_probe.py" in text
    assert "tests/test_hosting_r6_restart_healing.py" in text
    assert "tests/test_hosting_r7_acceptance.py" in text
    assert "native sandbox worker restart healing and cleanup" in text
