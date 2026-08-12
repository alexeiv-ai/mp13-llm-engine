"""Runtime mechanics behind the worker-neutral environment interface."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .contracts import EnvironmentRequest


class ManifestEnvironmentBuilder:
    """Portable baseline builder that materializes an immutable lock manifest."""

    def __init__(self, *, builder_id: str, runtime_kind: str) -> None:
        self.builder_id = str(builder_id)
        self.runtime_kind = str(runtime_kind)

    def build(
        self,
        *,
        request: EnvironmentRequest,
        destination: Path,
        package_lock: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        destination.mkdir(parents=True, exist_ok=True)
        manifest = {
            "contract": "hosting.materialized_environment.v1",
            "runtime_kind": self.runtime_kind,
            "platform": request.platform,
            "package_lock_digest": request.package_lock_digest,
            "artifacts": list(package_lock.get("artifacts") or []),
        }
        (destination / "environment.json").write_text(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
        return {"builder_id": self.builder_id, "manifest": "environment.json"}
