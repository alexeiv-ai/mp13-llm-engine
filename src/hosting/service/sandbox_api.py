"""Sandbox callback APIs exposed by the engine host service."""
from __future__ import annotations

from typing import Any, Dict, Optional


class SandboxApiMixin:
    @staticmethod
    def _sandbox_callback_result(result: Dict[str, Any], *, callback_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        payload = dict(result or {})
        if isinstance(callback_context, dict) and callback_context:
            payload["callback_context"] = dict(callback_context)
        return payload

    def sandbox_fs_read_text(
        self,
        *,
        engine_id: str,
        root_id: str,
        relative_path: str,
        encoding: str = "utf-8",
        callback_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return self._sandbox_callback_result(
            self._sandbox_fs_for_engine(engine_id).read_text(
                root_id=root_id,
                relative_path=relative_path,
                encoding=encoding,
            ),
            callback_context=callback_context,
        )

    def sandbox_fs_write_text(
        self,
        *,
        engine_id: str,
        root_id: str,
        relative_path: str,
        text: str,
        encoding: str = "utf-8",
        create_parents: bool = True,
        callback_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return self._sandbox_callback_result(
            self._sandbox_fs_for_engine(engine_id).write_text(
                root_id=root_id,
                relative_path=relative_path,
                text=text,
                encoding=encoding,
                create_parents=create_parents,
            ),
            callback_context=callback_context,
        )

    def sandbox_fs_mkdir(
        self,
        *,
        engine_id: str,
        root_id: str,
        relative_path: str,
        parents: bool = True,
        exist_ok: bool = True,
        callback_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return self._sandbox_callback_result(
            self._sandbox_fs_for_engine(engine_id).mkdir(
                root_id=root_id,
                relative_path=relative_path,
                parents=parents,
                exist_ok=exist_ok,
            ),
            callback_context=callback_context,
        )

    def sandbox_fs_list(
        self,
        *,
        engine_id: str,
        root_id: str,
        relative_path: Optional[str] = None,
        callback_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return self._sandbox_callback_result(
            self._sandbox_fs_for_engine(engine_id).list_dir(root_id=root_id, relative_path=relative_path),
            callback_context=callback_context,
        )

    def sandbox_fs_stat(
        self,
        *,
        engine_id: str,
        root_id: str,
        relative_path: Optional[str] = None,
        callback_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return self._sandbox_callback_result(
            self._sandbox_fs_for_engine(engine_id).stat(root_id=root_id, relative_path=relative_path),
            callback_context=callback_context,
        )

    def sandbox_http_fetch(
        self,
        *,
        engine_id: str,
        url: str,
        method: str = "GET",
        headers: Optional[Dict[str, str]] = None,
        body_b64: str = "",
        timeout_seconds: float = 30.0,
        max_response_bytes: int = 1024 * 1024,
        callback_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        out = self._sandbox_http_for_engine(engine_id).fetch(
            url=url,
            method=method,
            headers=headers,
            body_b64=body_b64,
            timeout_seconds=timeout_seconds,
            max_response_bytes=max_response_bytes,
        )
        return self._sandbox_callback_result(
            {"engine_id": str(engine_id or ""), **dict(out or {})},
            callback_context=callback_context,
        )

