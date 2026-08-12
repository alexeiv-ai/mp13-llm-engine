"""Canonical unified hosting configuration for v3 tests."""
from pathlib import Path

from hosting.hosting_configuration import parse_hosting_configuration
from mp13_engine.mp13_config_paths import resolve_config_paths


def hosting_configuration(root: Path):
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
        "control": {"authentication": {}, "roles": {}, "session_policy": {}, "audit": {}},
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
