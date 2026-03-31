from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional


RpcInvoker = Callable[[str, Dict[str, Any]], Dict[str, Any]]


@dataclass
class BrokeredFilesystemClient:
    """
    Worker-side adapter for brokered filesystem operations.

    This adapter is transport-agnostic on purpose: caller code provides
    an `rpc_invoke(command, payload)` callback. That keeps the broker client
    usable before the final worker<->host request path is fixed in code.
    """

    engine_id: str
    rpc_invoke: RpcInvoker

    def list_dir(self, *, root_id: str, relative_path: Optional[str] = None) -> Dict[str, Any]:
        return self.rpc_invoke(
            "sandbox-fs-list",
            {
                "engine_id": self.engine_id,
                "root_id": str(root_id or ""),
                "relative_path": relative_path,
            },
        )

    def read_text(self, *, root_id: str, relative_path: str, encoding: str = "utf-8") -> Dict[str, Any]:
        return self.rpc_invoke(
            "sandbox-fs-read-text",
            {
                "engine_id": self.engine_id,
                "root_id": str(root_id or ""),
                "relative_path": str(relative_path or ""),
                "encoding": str(encoding or "utf-8"),
            },
        )

    def write_text(
        self,
        *,
        root_id: str,
        relative_path: str,
        text: str,
        encoding: str = "utf-8",
        create_parents: bool = True,
    ) -> Dict[str, Any]:
        return self.rpc_invoke(
            "sandbox-fs-write-text",
            {
                "engine_id": self.engine_id,
                "root_id": str(root_id or ""),
                "relative_path": str(relative_path or ""),
                "text": str(text or ""),
                "encoding": str(encoding or "utf-8"),
                "create_parents": bool(create_parents),
            },
        )

    def mkdir(
        self,
        *,
        root_id: str,
        relative_path: str,
        parents: bool = True,
        exist_ok: bool = True,
    ) -> Dict[str, Any]:
        return self.rpc_invoke(
            "sandbox-fs-mkdir",
            {
                "engine_id": self.engine_id,
                "root_id": str(root_id or ""),
                "relative_path": str(relative_path or ""),
                "parents": bool(parents),
                "exist_ok": bool(exist_ok),
            },
        )

    def stat(self, *, root_id: str, relative_path: Optional[str] = None) -> Dict[str, Any]:
        return self.rpc_invoke(
            "sandbox-fs-stat",
            {
                "engine_id": self.engine_id,
                "root_id": str(root_id or ""),
                "relative_path": relative_path,
            },
        )
