from pathlib import Path

import pytest

from mp13_engine.mp13_config_paths import (
    PathResolver,
    build_engine_init_payload,
    resolve_config_paths,
    resolve_engine_inputs,
    load_json_config,
    save_json_config,
)


def _resolver(tmp_path: Path) -> PathResolver:
    return PathResolver(
        cwd=tmp_path / "cwd",
        config_dir=tmp_path / "config",
        home_dir=tmp_path / "home",
        project_dir=tmp_path / "project",
        category_roots={"models": tmp_path / "models"},
    )


def test_base_model_local_name_resolves_under_models_root(tmp_path: Path) -> None:
    resolver = _resolver(tmp_path)

    resolved = resolve_engine_inputs({"base_model_path": "granite-3.3-2b-instruct"}, resolver)

    assert resolved["base_model_path"] == str((tmp_path / "models" / "granite-3.3-2b-instruct").resolve())


def test_explicit_hf_base_model_ref_is_preserved_until_engine_payload(tmp_path: Path) -> None:
    resolver = _resolver(tmp_path)

    resolved = resolve_engine_inputs({"base_model_path": "hf:microsoft/Phi-3-mini-4k-instruct"}, resolver)
    payload = build_engine_init_payload(resolved)

    assert resolved["base_model_path"] == "hf:microsoft/Phi-3-mini-4k-instruct"
    assert payload["base_model_name_or_path"] == "microsoft/Phi-3-mini-4k-instruct"


def test_top_level_hf_base_model_ref_is_not_rewritten_as_local_path(tmp_path: Path) -> None:
    config = {
        "category_dirs": {"models_root_dir": str(tmp_path / "models")},
        "base_model_path": "hf:meta-llama/Llama-2-7b-chat-hf",
    }

    resolved, _ = resolve_config_paths(config, cwd=tmp_path, config_path=tmp_path / "config.json")

    assert resolved["base_model_path"] == "hf:meta-llama/Llama-2-7b-chat-hf"


def test_hosting_roots_resolve_from_shared_category_model(tmp_path: Path) -> None:
    config_path = tmp_path / "config" / "mp13_config.json"
    config = {
        "category_dirs": {
            "hosting_root_dir": "@config/host-data",
            "packages_root_dir": "@config/package-data",
            "environments_root_dir": "@config/environment-data",
        }
    }

    resolved, resolver = resolve_config_paths(config, cwd=tmp_path, config_path=config_path)

    assert resolved["category_dirs"]["hosting_root_dir"] == str((config_path.parent / "host-data").resolve())
    assert resolver.resolve("@packages/artifacts") == str((config_path.parent / "package-data" / "artifacts").resolve())
    assert resolver.resolve("@environments/python") == str((config_path.parent / "environment-data" / "python").resolve())


@pytest.mark.parametrize("value", ["@missing/data", "@/data"])
def test_unknown_labels_fail_closed(tmp_path: Path, value: str) -> None:
    with pytest.raises(ValueError, match="unknown_path_label"):
        _resolver(tmp_path).resolve(value)


@pytest.mark.parametrize(
    "value,code",
    [
        ("@project/hosting", "persistent_root_anchor_invalid"),
        ("@config/../outside", "traversal_invalid"),
        ("C:/outside", "persistent_root_must_use_stable_anchor"),
    ],
)
def test_persistent_roots_reject_unsafe_values(tmp_path: Path, value: str, code: str) -> None:
    with pytest.raises(ValueError, match=code):
        resolve_config_paths(
            {"category_dirs": {"hosting_root_dir": value}},
            cwd=tmp_path,
            config_path=tmp_path / "config" / "mp13_config.json",
        )


def test_category_root_cycles_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="category_root_cycle"):
        resolve_config_paths(
            {
                "category_dirs": {
                    "models_root_dir": "@adapters/models",
                    "adapters_root_dir": "@models/adapters",
                }
            },
            cwd=tmp_path,
            config_path=tmp_path / "config" / "mp13_config.json",
        )


def test_persistent_roots_must_not_overlap(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="persistent_roots_overlap"):
        resolve_config_paths(
            {
                "category_dirs": {
                    "hosting_root_dir": "@config/hosting",
                    "packages_root_dir": "@config/hosting/packages",
                    "environments_root_dir": "@config/environments",
                }
            },
            cwd=tmp_path,
            config_path=tmp_path / "config" / "mp13_config.json",
        )


def test_logical_root_values_survive_save_and_load(tmp_path: Path) -> None:
    path = tmp_path / "mp13_config.json"
    config = {
        "category_dirs": {
            "hosting_root_dir": "@home/.mp13-llm/hosting",
            "packages_root_dir": "@home/.mp13-llm/packages",
            "environments_root_dir": "@home/.mp13-llm/environments",
        },
        "unrelated": {"preserved": True},
    }

    save_json_config(config, path)

    assert load_json_config(path) == config
