from pathlib import Path

from mp13_engine.mp13_config_paths import (
    PathResolver,
    build_engine_init_payload,
    resolve_config_paths,
    resolve_engine_inputs,
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
