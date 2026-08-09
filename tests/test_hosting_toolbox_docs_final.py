from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "src" / "hosting" / "HOSTED_TOOLBOX_CONTRACT.md"
WORKER = ROOT / "src" / "hosting" / "sandbox" / "TOOLBOX_WORKER.md"


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_durable_docs_contain_only_supported_runtime_vocabulary() -> None:
    forbidden = {
        "migration",
        "version-1",
        "legacy",
        "compatibility",
        "register_",
        "unregister_",
        "environment_descriptions",
        "old procedural",
        "breaking cutover",
    }
    for path in (CONTRACT, WORKER):
        text = _text(path).lower()
        assert not {term for term in forbidden if term in text}, path


def test_worker_links_normative_contract_and_covers_supported_architecture() -> None:
    text = _text(WORKER)
    prose = " ".join(text.split())
    assert "[Hosted Toolbox Definition Contract](../HOSTED_TOOLBOX_CONTRACT.md)" in text
    for section in [
        "## Runtime model",
        "## Definition planning",
        "## Template and custom package environments",
        "## Bundle and worker startup",
        "## Candidate rollout and active routing",
        "## Durable apply and recovery",
        "## Execution, gates, callbacks, and cancellation",
        "## Projections and authorization",
        "## Maintenance and garbage collection",
    ]:
        assert section in text
    for guarantee in [
        "the complete previous revision or the complete new revision",
        "Candidates are explicitly non-routable.",
        "system_site_packages=False",
        "--no-index --no-deps",
        "The host interpreter and another venv are never dependency fallbacks.",
        "The version-2 snapshot is the routing source during reconciliation.",
        "From publication through cleanup the operation is non-cancellable",
        "Direct engine-ID diagnostics are an internal/operator surface.",
    ]:
        assert guarantee in prose


def test_worker_source_links_resolve() -> None:
    targets = re.findall(r"\[[^]]+\]\(([^)]+\.py)\)", _text(WORKER))
    assert len(targets) >= 15
    missing = [target for target in targets if not (WORKER.parent / target).resolve().is_file()]
    assert missing == []


def test_contract_and_worker_agree_on_core_runtime_boundaries() -> None:
    contract = " ".join(_text(CONTRACT).split())
    worker = " ".join(_text(WORKER).split())
    for phrase in [
        "`core` and `py-compute`",
        "expected active revision",
        "approval reference",
        "empty definition",
        "publication",
        "operator",
    ]:
        assert phrase.lower() in contract.lower()
        assert phrase.lower() in worker.lower()
