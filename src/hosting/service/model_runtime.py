"""Read-only bounded model-runtime projection."""
from __future__ import annotations

import time
from typing import Any

from ..model_runtime_contract import ModelRuntimeStatus


class ModelRuntimeMixin:
    def model_runtime_status(self) -> dict[str, Any]:
        identity = getattr(self, "_model_runtime_identity", None)
        if identity is None:
            return ModelRuntimeStatus(
                state="unavailable",
                code="model_runtime_unconfigured",
                summary="The exclusive model runtime is not configured on this host.",
                python_abi=None,
                platform=None,
                engine_artifact_digest=None,
                complete_lock_digest=None,
                optional_package_set=None,
                materialization_revision=None,
                updated_at_ms=int(time.time() * 1000),
            ).to_dict()
        ready = bool(identity.verified)
        return ModelRuntimeStatus(
            state="ready" if ready else "degraded",
            code="model_runtime_ready" if ready else "model_runtime_verification_failed",
            summary=(
                "The exclusive model runtime is verified and ready for authorized model operations."
                if ready
                else "The exclusive model runtime did not pass complete identity verification."
            ),
            python_abi=identity.python_abi,
            platform=identity.platform,
            engine_artifact_digest=identity.engine_artifact_digest,
            complete_lock_digest=identity.complete_lock_digest,
            optional_package_set=identity.optional_package_set,
            materialization_revision=identity.materialization_revision,
            updated_at_ms=identity.updated_at_ms,
        ).to_dict()


__all__ = ["ModelRuntimeMixin"]
