"""Dependency-light native CI probe for the hosting target detector."""
from __future__ import annotations

import argparse
import importlib.util
import platform
import sys
from pathlib import Path


def _target_module():
    path = Path(__file__).resolve().parents[1] / "src" / "hosting" / "toolbox" / "target.py"
    spec = importlib.util.spec_from_file_location("hosting_native_target_probe", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("target_module_load_failed")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-target", required=True)
    args = parser.parse_args()

    target_module = _target_module()
    target = target_module.detect_current_toolbox_target()
    if target.name != args.expected_target:
        raise AssertionError(
            f"detected target {target.name!r} != expected {args.expected_target!r}; "
            f"system={platform.system()!r} machine={platform.machine()!r}"
        )
    if not target.compatible_tags or not target.compatible_tags[0].startswith("cp312-"):
        raise AssertionError("current CPython ABI is not first in sys_tags")

    from pydantic_core import _pydantic_core

    extension_path = Path(_pydantic_core.__file__ or "")
    if not extension_path.is_file() or extension_path.suffix.lower() in {".py", ".pyc"}:
        raise AssertionError(f"pydantic-core did not import a native extension: {extension_path}")
    print(target.to_dict())
    print({"native_extension": str(extension_path), "machine": platform.machine()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
