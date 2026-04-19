"""Load staged toolbox bundles from manifest files."""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import Any, Dict

from mp13_engine.mp13_toolbox import Toolbox


def load_toolbox_from_manifest(manifest_path: Path) -> tuple[Toolbox, Dict[str, Any]]:
    manifest_file = Path(manifest_path).expanduser().resolve()
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("toolbox_manifest_invalid")
    bundle_root = manifest_file.parent
    files_root = (bundle_root / "files").resolve()
    if str(files_root) not in sys.path:
        sys.path.insert(0, str(files_root))
    intrinsic_tool_names = [
        str(item or "").strip()
        for item in list(manifest.get("intrinsic_tool_names") or [])
        if str(item or "").strip()
    ]
    toolbox = Toolbox()
    hidden_user_tools = [
        str(item or "").strip()
        for item in list(manifest.get("hidden_tool_names") or [])
        if str(item or "").strip()
    ]
    if intrinsic_tool_names:
        ok, msg = toolbox.add_tool_callable(
            intrinsic_tool_names,
            is_intrinsic=True,
            include_guides=bool(manifest.get("with_intrinsic_guides", False)),
            activate=True,
        )
        if not ok:
            raise ValueError(str(msg or "intrinsic_registration_failed"))
        active_intrinsic = [
            str(item or "").strip()
            for item in list(manifest.get("active_intrinsic_tool_names") or [])
            if str(item or "").strip()
        ]
        hidden_intrinsic = [
            str(item or "").strip()
            for item in list(manifest.get("hidden_intrinsic_tool_names") or [])
            if str(item or "").strip()
        ]
        if active_intrinsic:
            toolbox.active_intrinsic_tool_names = [
                name for name in active_intrinsic if name in toolbox.intrinsic_tools
            ]
        if hidden_intrinsic:
            toolbox.hidden_intrinsic_tool_names = [
                name for name in hidden_intrinsic if name in toolbox.intrinsic_tools
            ]
    for item in list(manifest.get("auto_tools") or []):
        auto_meta = dict(item or {})
        module_name = str(auto_meta.get("module_name") or "").strip()
        callable_name = str(auto_meta.get("callable_name") or "").strip()
        if not module_name:
            raise ValueError("auto_tool_module_name_required")
        if not callable_name:
            raise ValueError("auto_tool_callable_name_required")
        module = importlib.import_module(module_name)
        ok, msg = toolbox.add_tool_callable(
            callable_name,
            search_scope=dict(vars(module)),
            activate=bool(auto_meta.get("activate", True)),
            guide_content=dict(auto_meta.get("guide_content") or {}) or None,
            guide_description=str(auto_meta.get("guide_description") or "").strip() or None,
        )
        if not ok:
            raise ValueError(str(msg or "auto_tool_registration_failed"))
        tool_def = toolbox.get_tool(callable_name)
        if tool_def is not None:
            tool_def["callback_signature"] = dict(auto_meta.get("callback_signature") or {}) or None
    for item in list(manifest.get("tools") or []):
        tool_meta = dict(item or {})
        entrypoint = str(tool_meta.get("entrypoint") or "").strip()
        if ":" not in entrypoint:
            raise ValueError(f"tool_entrypoint_invalid:{entrypoint}")
        module_name, attr_name = entrypoint.split(":", 1)
        module = importlib.import_module(module_name)
        implementation = getattr(module, attr_name)
        ok, msg = toolbox.add_tool_external(
            tool_definition=dict(tool_meta.get("definition") or {}),
            implementation=implementation,
            activate=True,
            allow_override=False,
        )
        if not ok:
            raise ValueError(str(msg or "tool_registration_failed"))
        tool_name = str(dict(tool_meta.get("definition") or {}).get("function", {}).get("name") or "").strip()
        tool_def = toolbox.get_tool(tool_name) if tool_name else None
        if tool_def is not None:
            tool_def["callback_signature"] = dict(tool_meta.get("callback_signature") or {}) or None
    if hidden_user_tools:
        toolbox.hidden_tool_names = [
            name for name in hidden_user_tools if name in toolbox.tools
        ]
    return toolbox, manifest
