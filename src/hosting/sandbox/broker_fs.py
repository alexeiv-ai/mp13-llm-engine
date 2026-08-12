from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from .policy import SandboxFsRule, WorkerSandboxPolicy


class BrokeredFsError(PermissionError):
    pass


def _safe_root_id(value: Optional[str]) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    return raw


def _resolve_root(path: str) -> Path:
    return Path(str(path or "").strip()).expanduser().resolve()


def _normalize_relative_path(value: Optional[str]) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    return raw.replace("\\", "/")


def _contains_access(rule: SandboxFsRule, access: str) -> bool:
    return str(access or "").strip().lower() in {str(x) for x in list(rule.access or [])}


@dataclass
class BrokeredFilesystem:
    policy: WorkerSandboxPolicy

    def _rule_for_root(self, root_id: str) -> SandboxFsRule:
        rid = _safe_root_id(root_id)
        if not rid:
            raise BrokeredFsError("root_id_required")
        for rule in list(self.policy.filesystem_rules or []):
            if str(rule.root_id or "").strip() == rid:
                return rule
        raise BrokeredFsError("unknown_root_id")

    def _resolve_path(self, root_id: str, relative_path: Optional[str]) -> tuple[SandboxFsRule, Path]:
        rule = self._rule_for_root(root_id)
        root = _resolve_root(rule.path)
        rel = _normalize_relative_path(relative_path)
        target = (root / rel).resolve() if rel else root
        try:
            target.relative_to(root)
        except Exception as exc:
            raise BrokeredFsError("path_traversal_denied") from exc
        return rule, target

    def list_dir(self, *, root_id: str, relative_path: Optional[str] = None) -> Dict[str, Any]:
        rule, target = self._resolve_path(root_id, relative_path)
        if not _contains_access(rule, "read"):
            raise BrokeredFsError("read_access_denied")
        if not target.exists():
            raise FileNotFoundError(str(target))
        if not target.is_dir():
            raise NotADirectoryError(str(target))
        rows = []
        for child in sorted(target.iterdir(), key=lambda p: p.name.lower()):
            rows.append(
                {
                    "name": child.name,
                    "is_dir": child.is_dir(),
                    "is_file": child.is_file(),
                    "size": int(child.stat().st_size) if child.exists() and child.is_file() else None,
                }
            )
        return {"root_id": root_id, "path": str(target), "entries": rows}

    def read_text(self, *, root_id: str, relative_path: str, encoding: str = "utf-8") -> Dict[str, Any]:
        rule, target = self._resolve_path(root_id, relative_path)
        if not _contains_access(rule, "read"):
            raise BrokeredFsError("read_access_denied")
        return {
            "root_id": root_id,
            "path": str(target),
            "text": target.read_text(encoding=encoding),
        }

    def write_text(
        self,
        *,
        root_id: str,
        relative_path: str,
        text: str,
        encoding: str = "utf-8",
        create_parents: bool = True,
    ) -> Dict[str, Any]:
        rule, target = self._resolve_path(root_id, relative_path)
        if not _contains_access(rule, "write"):
            raise BrokeredFsError("write_access_denied")
        if create_parents:
            target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(str(text), encoding=encoding)
        return {"root_id": root_id, "path": str(target), "bytes_written": len(str(text).encode(encoding))}

    def mkdir(self, *, root_id: str, relative_path: str, parents: bool = True, exist_ok: bool = True) -> Dict[str, Any]:
        rule, target = self._resolve_path(root_id, relative_path)
        if not _contains_access(rule, "write"):
            raise BrokeredFsError("write_access_denied")
        target.mkdir(parents=parents, exist_ok=exist_ok)
        return {"root_id": root_id, "path": str(target), "created": True}

    def stat(self, *, root_id: str, relative_path: Optional[str] = None) -> Dict[str, Any]:
        rule, target = self._resolve_path(root_id, relative_path)
        if not _contains_access(rule, "read"):
            raise BrokeredFsError("read_access_denied")
        st = target.stat()
        return {
            "root_id": root_id,
            "path": str(target),
            "exists": True,
            "is_dir": target.is_dir(),
            "is_file": target.is_file(),
            "size": int(st.st_size),
            "mode": stat_mode_string(st.st_mode),
        }


def stat_mode_string(mode: int) -> str:
    return oct(int(mode) & 0o777)
