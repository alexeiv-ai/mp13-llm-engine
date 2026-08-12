"""Canonical unified hosting configuration for v3 tests."""
from pathlib import Path
import json
from collections.abc import Mapping

from hosting.hosting_configuration import parse_hosting_configuration
from mp13_engine.mp13_config_paths import resolve_config_paths


def hosting_configuration(
    root: Path,
    *,
    require_auth: bool = False,
    connectivity_mode: str = "local_only",
    endpoint_mode: str = "exclusive",
    lifecycle: dict | None = None,
    claims: dict | None = None,
):
    root = Path(root).resolve()
    _, resolver = resolve_config_paths(
        {"category_dirs": {
            "hosting_root_dir": "@config/host",
            "packages_root_dir": "@config/packages",
            "environments_root_dir": "@config/environments",
        }}, cwd=root, config_path=root / "mp13_config.json", project_root=root,
    )
    return parse_hosting_configuration({
        "contract": "hosting.configuration.v3",
        "control": {
            "authentication": {
                "require_auth": require_auth,
                "connectivity_mode": connectivity_mode,
                "endpoint_mode": endpoint_mode,
            },
            "roles": {},
            "session_policy": {},
            "audit": {},
            "lifecycle": dict(lifecycle or {}),
            "claims": dict(claims or {}),
        },
        "package_management": {
            "artifact_root": "@packages/artifacts", "lock_root": "@packages/locks",
            "sources": {}, "credentials": {},
            "dependency_policy": {
                "policy_id": "default", "revision": 1, "allowed_source_ids": [],
                "allowed_platforms": ["*"], "allowed_runtimes": ["python", "javascript"],
                "max_artifact_bytes": 67108864, "require_sha256": True, "optional_verifier": None,
            }, "verification": {"hash_algorithm": "sha256"},
        },
        "environment_management": {
            "environment_root": "@environments", "scratch_root": "@hosting/scratch",
            "retention": {}, "cache": {},
        },
    }, resolver)


def write_hosting_configuration(
    root: Path,
    *,
    require_auth: bool = False,
    connectivity_mode: str = "local_only",
    endpoint_mode: str = "exclusive",
    lifecycle: dict | None = None,
    claims: dict | None = None,
) -> Path:
    root = Path(root).resolve()
    config = root / "mp13_config.json"
    config.write_text(json.dumps({"category_dirs": {
        "hosting_root_dir": "@config/host", "packages_root_dir": "@config/packages",
        "environments_root_dir": "@config/environments",
    }}), encoding="utf-8")
    authority = root / "hosting" / "hosting_config.json"
    authority.parent.mkdir(parents=True, exist_ok=True)
    model = hosting_configuration(
        root,
        require_auth=require_auth,
        connectivity_mode=connectivity_mode,
        endpoint_mode=endpoint_mode,
        lifecycle=lifecycle,
        claims=claims,
    )
    def plain(value):
        if isinstance(value, Mapping):
            return {str(key): plain(item) for key, item in value.items()}
        if isinstance(value, tuple):
            return [plain(item) for item in value]
        return value
    authority.write_text(json.dumps({
        "contract": model.contract,
        "control": plain(model.control),
        "package_management": plain(model.package_management),
        "environment_management": plain(model.environment_management),
    }), encoding="utf-8")
    return config
