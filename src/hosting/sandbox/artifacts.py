"""Shared host-provisioned artifact I/O helpers for hosted sandboxes."""
from __future__ import annotations

import base64
import io
import json
import shutil
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


def artifact_safe_name(value: Any, *, fallback: str) -> str:
    raw = str(value or "").strip() or fallback
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in raw)
    return safe.strip("._") or fallback


def artifact_safe_relpath(value: Any, *, fallback: str) -> str:
    raw = str(value or "").strip().replace("\\", "/").strip("/")
    if not raw:
        raw = fallback
    parts = [artifact_safe_name(part, fallback="") for part in raw.split("/") if artifact_safe_name(part, fallback="")]
    if not parts:
        parts = [artifact_safe_name(fallback, fallback="artifact")]
    return "/".join(parts)


def artifact_ref_parts(ref: str) -> Optional[tuple[str, str]]:
    value = str(ref or "").strip().replace("\\", "/")
    if not value.startswith("@"):
        return None
    alias, _, rel = value[1:].partition("/")
    alias = alias.strip()
    rel = rel.strip("/")
    if not alias or not rel:
        return None
    path = Path(rel)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        return None
    return alias, rel


def artifact_path_from_ref(ref: str, *, roots: Dict[str, Path]) -> Optional[Path]:
    parts = artifact_ref_parts(ref)
    if parts is None:
        return None
    alias, rel = parts
    root = roots.get(alias)
    if root is None:
        return None
    path = (root / rel).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError:
        return None
    return path


def artifact_has_mask(value: str) -> bool:
    text = str(value or "")
    return any(ch in text for ch in ("*", "?", "["))


def inline_artifact_bytes(row: Dict[str, Any]) -> Optional[bytes]:
    if "base64" in row:
        return base64.b64decode(str(row.get("base64") or ""), validate=True)
    if "text" in row:
        encoding = str(row.get("encoding") or "utf-8").strip() or "utf-8"
        return str(row.get("text") or "").encode(encoding)
    if "data" in row:
        data = row.get("data")
        if isinstance(data, bytes):
            return data
        encoding = str(row.get("encoding") or "utf-8").strip() or "utf-8"
        return str(data or "").encode(encoding)
    if str(row.get("kind") or row.get("mode") or "").strip().lower() == "inline":
        return b""
    return None


def _bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _optional(row: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in dict(row or {}).items() if value is not None and value != ""}


@dataclass(frozen=True)
class HostedArtifactRow:
    """Stable serializable artifact row used by hosted workflow requests."""

    row: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.row or {})

    @classmethod
    def inline_input(
        cls,
        *,
        name: str,
        text: Optional[str] = None,
        base64_data: str = "",
        data: Any = None,
        filename: str = "",
        media_type: str = "text/plain",
        encoding: str = "utf-8",
        **metadata: Any,
    ) -> "HostedArtifactRow":
        row = {
            "name": name,
            "kind": "inline",
            "filename": filename or f"{artifact_safe_name(name, fallback='input')}.txt",
            "media_type": media_type,
            "encoding": encoding,
            **dict(metadata or {}),
        }
        if base64_data:
            row["base64"] = base64_data
        elif data is not None:
            row["data"] = data
        else:
            row["text"] = "" if text is None else text
        return cls(_optional(row))

    @classmethod
    def inline_zip_input(
        cls,
        *,
        name: str,
        base64_data: str,
        filename: str = "",
        **metadata: Any,
    ) -> "HostedArtifactRow":
        return cls(
            _optional(
                {
                    "name": name,
                    "kind": "inline",
                    "filename": filename or f"{artifact_safe_name(name, fallback='project')}.zip",
                    "media_type": "application/zip",
                    "encoding": "zip",
                    "base64": base64_data,
                    **dict(metadata or {}),
                }
            )
        )

    @classmethod
    def ref_input(
        cls,
        *,
        name: str,
        ref: str,
        filename: str = "",
        media_type: str = "application/octet-stream",
        **metadata: Any,
    ) -> "HostedArtifactRow":
        return cls(_optional({"name": name, "kind": "ref", "ref": ref, "filename": filename, "media_type": media_type, **dict(metadata or {})}))

    @classmethod
    def masked_ref_input(
        cls,
        *,
        name: str,
        ref: str,
        path_mask: str = "*",
        recursive: bool = True,
        media_type: str = "application/octet-stream",
        **metadata: Any,
    ) -> "HostedArtifactRow":
        return cls(
            _optional(
                {
                    "name": name,
                    "kind": "ref",
                    "ref": ref,
                    "path_mask": path_mask or "*",
                    "recursive": bool(recursive),
                    "media_type": media_type,
                    **dict(metadata or {}),
                }
            )
        )

    @classmethod
    def file_output(
        cls,
        *,
        name: str,
        filename: str = "",
        media_type: str = "application/octet-stream",
        **metadata: Any,
    ) -> "HostedArtifactRow":
        return cls(
            _optional(
                {
                    "name": name,
                    "kind": "ref",
                    "filename": filename or f"{artifact_safe_name(name, fallback='output')}.bin",
                    "media_type": media_type,
                    **dict(metadata or {}),
                }
            )
        )

    @classmethod
    def host_takeover_output(
        cls,
        *,
        name: str,
        ref: str,
        filename: str = "",
        media_type: str = "application/octet-stream",
        **metadata: Any,
    ) -> "HostedArtifactRow":
        return cls(
            _optional(
                {
                    "name": name,
                    "kind": "ref",
                    "ref": ref,
                    "filename": filename or f"{artifact_safe_name(name, fallback='output')}.bin",
                    "media_type": media_type,
                    "host_takeover": True,
                    **dict(metadata or {}),
                }
            )
        )

    @classmethod
    def producer_owned_output(
        cls,
        *,
        name: str,
        ref: str,
        filename: str = "",
        media_type: str = "application/octet-stream",
        **metadata: Any,
    ) -> "HostedArtifactRow":
        return cls(
            _optional(
                {
                    "name": name,
                    "kind": "ref",
                    "ref": ref,
                    "filename": filename or f"{artifact_safe_name(name, fallback='output')}.bin",
                    "media_type": media_type,
                    "ownership": "producer",
                    **dict(metadata or {}),
                }
            )
        )

    @classmethod
    def inline_zip_output(
        cls,
        *,
        name: str,
        ref: str = "",
        path_mask: str = "*",
        recursive: bool = True,
        filename: str = "",
        **metadata: Any,
    ) -> "HostedArtifactRow":
        return cls(
            _optional(
                {
                    "name": name,
                    "kind": "ref",
                    "ref": ref,
                    "path_mask": path_mask or "*",
                    "recursive": bool(recursive),
                    "filename": filename or f"{artifact_safe_name(name, fallback='artifacts')}.zip",
                    "media_type": "application/zip",
                    "export_inline_zip": True,
                    **dict(metadata or {}),
                }
            )
        )


def artifact_inline_input(**kwargs: Any) -> Dict[str, Any]:
    return HostedArtifactRow.inline_input(**kwargs).to_dict()


def artifact_inline_zip_input(**kwargs: Any) -> Dict[str, Any]:
    return HostedArtifactRow.inline_zip_input(**kwargs).to_dict()


def artifact_ref_input(**kwargs: Any) -> Dict[str, Any]:
    return HostedArtifactRow.ref_input(**kwargs).to_dict()


def artifact_masked_ref_input(**kwargs: Any) -> Dict[str, Any]:
    return HostedArtifactRow.masked_ref_input(**kwargs).to_dict()


def artifact_file_output(**kwargs: Any) -> Dict[str, Any]:
    return HostedArtifactRow.file_output(**kwargs).to_dict()


def artifact_host_takeover_output(**kwargs: Any) -> Dict[str, Any]:
    return HostedArtifactRow.host_takeover_output(**kwargs).to_dict()


def artifact_producer_owned_output(**kwargs: Any) -> Dict[str, Any]:
    return HostedArtifactRow.producer_owned_output(**kwargs).to_dict()


def artifact_inline_zip_output(**kwargs: Any) -> Dict[str, Any]:
    return HostedArtifactRow.inline_zip_output(**kwargs).to_dict()


def _zip_entries(data: bytes) -> list[tuple[str, bytes]]:
    out: list[tuple[str, bytes]] = []
    with zipfile.ZipFile(io.BytesIO(data), "r") as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            rel = Path(str(info.filename).replace("\\", "/"))
            if rel.is_absolute() or any(part in {"", ".", ".."} for part in rel.parts):
                raise ValueError(f"inline_zip_path_invalid:{info.filename}")
            out.append(("/".join(rel.parts), zf.read(info)))
    return out


class HostedArtifactManager:
    """Lean host-side artifact staging and collection.

    Inline artifacts are receiver-managed and can carry many files as zip data.
    Ref artifacts are producer-managed unless an output declaration explicitly
    asks the host to take over by omitting `ref` or setting `host_takeover`.
    """

    def __init__(self, *, artifact_root: Path, artifact_roots: Optional[Dict[str, Path]] = None) -> None:
        self.artifact_root = Path(artifact_root).expanduser().resolve()
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        self.roots: Dict[str, Path] = {"artifacts": (self.artifact_root / "objects").resolve()}
        for alias, path in dict(artifact_roots or {}).items():
            key = artifact_safe_name(alias, fallback="")
            if key:
                self.roots[key] = Path(path).expanduser().resolve()
        for root in self.roots.values():
            root.mkdir(parents=True, exist_ok=True)

    def path_from_ref(self, ref: str) -> Optional[Path]:
        return artifact_path_from_ref(ref, roots=self.roots)

    def paths_from_ref(self, row: Dict[str, Any]) -> tuple[Optional[Path], list[Path]]:
        ref = str(row.get("ref") or "").strip()
        path_mask = str(row.get("path_mask") or row.get("mask") or "").strip()
        recursive = _bool(row.get("recursive", False))
        parts = artifact_ref_parts(ref)
        if parts is None:
            return None, []
        alias, rel = parts
        root = self.roots.get(alias)
        if root is None:
            return None, []
        if path_mask:
            base = self.path_from_ref(ref)
            if base is None or not base.exists() or not base.is_dir():
                return base, []
            matches = list(base.rglob(path_mask) if recursive else base.glob(path_mask))
            return base, sorted(path.resolve() for path in matches if path.is_file())
        if artifact_has_mask(rel):
            pattern = rel
            if recursive and not pattern.startswith("**/"):
                pattern = f"**/{pattern}"
            matches = list(root.glob(pattern))
            return root, sorted(path.resolve() for path in matches if path.is_file())
        path = self.path_from_ref(ref)
        return path, [path] if path is not None and path.exists() and path.is_file() else []

    def prepare(
        self,
        *,
        request: Dict[str, Any],
        request_id: str,
    ) -> Dict[str, Any]:
        inputs = []
        outputs = []
        child_inputs: Dict[str, str] = {}
        child_outputs: Dict[str, str] = {}
        run_root = (self.artifact_root / "runs" / artifact_safe_name(request_id, fallback="request")).resolve()
        input_root = run_root / "inputs"
        output_root = run_root / "outputs"
        input_root.mkdir(parents=True, exist_ok=True)
        output_root.mkdir(parents=True, exist_ok=True)
        self._write_recovery_manifest(run_root, request=request, request_id=request_id)
        for index, spec in enumerate(list(request.get("artifact_inputs") or [])):
            row = dict(spec or {})
            name = artifact_safe_name(row.get("name"), fallback=f"input_{index}")
            inline_bytes = inline_artifact_bytes(row)
            source = None
            matches: list[Path] = []
            if inline_bytes is None:
                source, matches = self.paths_from_ref(row)
            inline_zip = inline_bytes is not None and (
                _bool(row.get("zip")) or str(row.get("format") or row.get("encoding") or "").strip().lower() in {"zip", "application/zip"}
            )
            if inline_bytes is None and not matches:
                raise ValueError(f"artifact_input_unavailable:{name}")
            masked_input = inline_zip or (
                inline_bytes is None
                and bool(str(row.get("path_mask") or row.get("mask") or "").strip() or artifact_has_mask(str(row.get("ref") or "")))
            )
            filename = artifact_safe_name(row.get("filename") or (matches[0].name if matches else name), fallback="artifact.bin")
            target = ((input_root / name) if masked_input else (input_root / name / filename)).resolve()
            target.parent.mkdir(parents=True, exist_ok=True)
            files = []
            if inline_zip:
                target.mkdir(parents=True, exist_ok=True)
                for rel_text, data in _zip_entries(inline_bytes or b""):
                    rel = Path(rel_text)
                    out_path = (target / rel).resolve()
                    try:
                        out_path.relative_to(target)
                    except ValueError as exc:
                        raise ValueError(f"artifact_input_path_invalid:{name}") from exc
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    out_path.write_bytes(data)
                    files.append({"path": str(out_path), "relative_path": rel_text})
            elif inline_bytes is not None:
                target.write_bytes(inline_bytes)
            elif masked_input:
                target.mkdir(parents=True, exist_ok=True)
                base = source if source is not None and source.is_dir() else (source.parent if source is not None else matches[0].parent)
                for match in matches:
                    try:
                        rel = match.relative_to(base)
                    except ValueError:
                        rel = Path(match.name)
                    out_path = (target / rel).resolve()
                    try:
                        out_path.relative_to(target)
                    except ValueError as exc:
                        raise ValueError(f"artifact_input_path_invalid:{name}") from exc
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(match, out_path)
                    files.append({"source": str(match), "path": str(out_path), "relative_path": str(rel).replace("\\", "/")})
            else:
                shutil.copyfile(matches[0], target)
                files = [{"source": str(matches[0]), "path": str(target), "relative_path": filename}]
            child_inputs[name] = str(target)
            inputs.append(
                {
                    "name": name,
                    "kind": "inline" if inline_bytes is not None else "ref",
                    "ref": str(row.get("ref") or "") or None,
                    "path": str(target),
                    "filename": filename,
                    "path_mask": str(row.get("path_mask") or row.get("mask") or "").strip() or None,
                    "recursive": _bool(row.get("recursive", False)),
                    "zip": bool(inline_zip),
                    "files": files if inline_zip or inline_bytes is None else [],
                    "media_type": str(row.get("media_type") or row.get("content_type") or "application/octet-stream"),
                    "encoding": str(row.get("encoding") or "").strip() or None,
                    "max_bytes": row.get("max_bytes"),
                    "count": row.get("count"),
                    "ttl": row.get("ttl"),
                    "lifetime": row.get("lifetime"),
                    "expires_at": row.get("expires_at"),
                }
            )
        for index, spec in enumerate(list(request.get("artifact_outputs") or [])):
            row = dict(spec or {})
            name = artifact_safe_name(row.get("name"), fallback=f"output_{index}")
            filename = artifact_safe_name(row.get("filename"), fallback=f"{name}.bin")
            kind = str(row.get("kind") or row.get("mode") or ("inline" if _bool(row.get("inline")) else "ref")).strip().lower() or "ref"
            if kind == "inline":
                outputs.append(self._output_spec(row, name=name, kind="inline", filename=filename))
                continue
            path_mask = str(row.get("path_mask") or row.get("mask") or "").strip()
            target = ((output_root / name) if path_mask else (output_root / name / filename)).resolve()
            try:
                target.relative_to(output_root)
            except ValueError as exc:
                raise ValueError(f"artifact_output_path_invalid:{name}") from exc
            if path_mask:
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
            child_outputs[name] = str(target)
            output_ref = str(row.get("ref") or "").strip() or None
            if output_ref and self.path_from_ref(output_ref) is None:
                raise ValueError(f"artifact_output_ref_invalid:{name}")
            outputs.append(self._output_spec(row, name=name, kind="ref", filename=filename, path=str(target), ref=output_ref))
        return {
            "run_root": str(run_root),
            "roots": {f"@{key}": str(value) for key, value in sorted(self.roots.items())},
            "inputs": inputs,
            "outputs": outputs,
            "child_context": {"inputs": child_inputs, "outputs": child_outputs},
        }

    @staticmethod
    def _write_recovery_manifest(run_root: Path, *, request: Dict[str, Any], request_id: str) -> None:
        row = dict(request or {})
        manifest = {
            "contract": "hosting.sandbox.artifact_recovery_manifest.v1",
            "request_id": artifact_safe_name(request_id, fallback="request"),
            "instance_id": str(row.get("instance_id") or "").strip() or None,
            "workflow_id": str(row.get("workflow_id") or "").strip() or None,
            "package_id": str(row.get("package_id") or "").strip() or None,
            "node_id": str(row.get("node_id") or "").strip() or None,
            "created_at": time.time(),
        }
        try:
            (run_root / "recovery_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            return

    @staticmethod
    def _read_recovery_manifest(run_root: Path) -> Dict[str, Any]:
        path = run_root / "recovery_manifest.json"
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
            return dict(row or {}) if isinstance(row, dict) else {}
        except Exception:
            return {}

    def _output_spec(
        self,
        row: Dict[str, Any],
        *,
        name: str,
        kind: str,
        filename: str,
        path: Optional[str] = None,
        ref: Optional[str] = None,
    ) -> Dict[str, Any]:
        ownership = str(row.get("ownership") or "").strip().lower()
        host_takeover = _bool(row.get("host_takeover")) or _bool(row.get("takeover")) or ownership in {"host", "host_takeover"}
        return {
            "name": name,
            "kind": kind,
            "filename": filename,
            "path": path,
            "ref": ref,
            "path_mask": str(row.get("path_mask") or row.get("mask") or "").strip() or None,
            "recursive": _bool(row.get("recursive", False)),
            "media_type": str(row.get("media_type") or row.get("content_type") or "application/octet-stream"),
            "encoding": str(row.get("encoding") or ("utf-8" if kind == "inline" else "")).strip() or None,
            "max_bytes": row.get("max_bytes"),
            "count": row.get("count"),
            "ttl": row.get("ttl"),
            "lifetime": row.get("lifetime"),
            "expires_at": row.get("expires_at"),
            "host_takeover": bool(host_takeover),
            "ownership": "host" if host_takeover or not ref else "producer",
            "export_inline_zip": _bool(row.get("export_inline_zip")) or _bool(row.get("export_zip_inline")),
        }

    def collect(
        self,
        context: Dict[str, Any],
        *,
        request_id: str,
        runtime_artifacts: Optional[list[Dict[str, Any]]] = None,
    ) -> list[Dict[str, Any]]:
        out = []
        for row in list(dict(context or {}).get("outputs") or []):
            spec = dict(row or {})
            if str(spec.get("kind") or "ref").strip().lower() == "inline":
                artifact = self._collect_inline(spec, runtime_artifacts=list(runtime_artifacts or []))
                if artifact is not None:
                    out.append(artifact)
                continue
            out.extend(self._collect_ref(spec, request_id=request_id))
        return out

    def _collect_inline(self, spec: Dict[str, Any], *, runtime_artifacts: list[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        match = next(
            (
                dict(item or {})
                for item in runtime_artifacts
                if isinstance(item, dict) and artifact_safe_name(item.get("name"), fallback="") == str(spec.get("name") or "")
            ),
            None,
        )
        if match is None:
            return None
        inline_bytes = inline_artifact_bytes(match)
        if inline_bytes is None:
            return None
        encoding = str(match.get("encoding") or spec.get("encoding") or "utf-8").strip() or "utf-8"
        artifact = {
            "name": str(spec.get("name") or "").strip() or None,
            "kind": "inline",
            "filename": str(match.get("filename") or spec.get("filename") or "").strip() or None,
            "media_type": str(match.get("media_type") or match.get("content_type") or spec.get("media_type") or "application/octet-stream"),
            "encoding": encoding,
            "size_bytes": len(inline_bytes),
        }
        if "base64" in match:
            artifact["base64"] = str(match.get("base64") or "")
        else:
            try:
                artifact["text"] = inline_bytes.decode(encoding)
            except UnicodeDecodeError:
                artifact["base64"] = base64.b64encode(inline_bytes).decode("ascii")
                artifact["encoding"] = "base64"
        return artifact

    def _collect_ref(self, spec: Dict[str, Any], *, request_id: str) -> list[Dict[str, Any]]:
        source = Path(str(spec.get("path") or "")).expanduser().resolve()
        path_mask = str(spec.get("path_mask") or "").strip()
        recursive = _bool(spec.get("recursive", False))
        if path_mask and source.exists() and source.is_dir():
            sources = sorted(path.resolve() for path in (source.rglob(path_mask) if recursive else source.glob(path_mask)) if path.is_file())
        else:
            sources = [source] if source.exists() and source.is_file() else []
        if not sources:
            return []
        if _bool(spec.get("export_inline_zip")):
            return [self._export_zip_inline(spec, base=source, sources=sources)]
        out = []
        for source_file in sources:
            out.append(self._collect_ref_file(spec, request_id=request_id, source=source, source_file=source_file))
        return out

    def _collect_ref_file(self, spec: Dict[str, Any], *, request_id: str, source: Path, source_file: Path) -> Dict[str, Any]:
        path_mask = str(spec.get("path_mask") or "").strip()
        try:
            rel = source_file.relative_to(source) if source.is_dir() else Path(source_file.name)
        except ValueError:
            rel = Path(source_file.name)
        artifact_id = artifact_safe_name(f"{request_id}-{spec.get('name')}-{int(time.time() * 1000)}", fallback="artifact")
        filename = artifact_safe_name((str(rel).replace("\\", "_") if path_mask else spec.get("filename")) or source_file.name, fallback="artifact.bin")
        ref = str(spec.get("ref") or "").strip()
        host_takeover = _bool(spec.get("host_takeover")) or not ref
        if ref and not host_takeover:
            target_base = self.path_from_ref(ref)
            if target_base is None:
                raise ValueError(f"artifact_output_ref_invalid:{spec.get('name')}")
            target = (target_base / rel).resolve() if path_mask else target_base
            try:
                target.relative_to(target_base if path_mask else target.parent)
            except ValueError as exc:
                raise ValueError(f"artifact_output_ref_invalid:{spec.get('name')}") from exc
            rel_ref = str(rel).replace("\\", "/")
            out_ref = f"{ref.rstrip('/')}/{rel_ref}" if path_mask else ref
            ownership = "producer"
        else:
            rel_ref = str(rel).replace("\\", "/")
            out_ref = f"@artifacts/{artifact_id}/{filename}" if not path_mask else f"@artifacts/{artifact_id}/{rel_ref}"
            target = self.path_from_ref(out_ref)
            if target is None:
                raise ValueError(f"artifact_output_ref_invalid:{spec.get('name')}")
            ownership = "host"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_file, target)
        artifact = {
            "name": str(spec.get("name") or "").strip() or None,
            "kind": "ref",
            "ref": out_ref,
            "filename": filename,
            "media_type": str(spec.get("media_type") or "application/octet-stream"),
            "size_bytes": int(target.stat().st_size),
            "encoding": str(spec.get("encoding") or "").strip() or None,
        }
        if _bool(spec.get("host_takeover")):
            artifact["ownership"] = ownership
            artifact["host_takeover"] = ownership == "host"
        if path_mask:
            artifact["relative_path"] = str(rel).replace("\\", "/")
        return artifact

    def _export_zip_inline(self, spec: Dict[str, Any], *, base: Path, sources: list[Path]) -> Dict[str, Any]:
        raw = io.BytesIO()
        with zipfile.ZipFile(raw, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for source_file in sources:
                try:
                    rel = source_file.relative_to(base) if base.is_dir() else Path(source_file.name)
                except ValueError:
                    rel = Path(source_file.name)
                zf.write(source_file, "/".join(rel.parts))
        data = raw.getvalue()
        return {
            "name": str(spec.get("name") or "").strip() or None,
            "kind": "inline",
            "filename": artifact_safe_name(spec.get("filename") or f"{spec.get('name') or 'artifacts'}.zip", fallback="artifacts.zip"),
            "media_type": "application/zip",
            "encoding": "base64",
            "base64": base64.b64encode(data).decode("ascii"),
            "size_bytes": len(data),
            "ownership": "producer",
            "export_inline_zip": True,
        }

    def cleanup_run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        run_root = Path(str(dict(context or {}).get("run_root") or "")).expanduser()
        if not str(run_root):
            return {"status": "skipped", "reason": "run_root_missing"}
        resolved = run_root.resolve()
        try:
            resolved.relative_to((self.artifact_root / "runs").resolve())
        except ValueError:
            return {"status": "skipped", "reason": "run_root_outside_artifact_root"}
        if not resolved.exists():
            return {"status": "ok", "deleted": False}
        shutil.rmtree(resolved)
        return {"status": "ok", "deleted": True, "path": str(resolved)}

    def run_root_for_request(self, request_id: str) -> Path:
        return (self.artifact_root / "runs" / artifact_safe_name(request_id, fallback="request")).resolve()

    def recovery_candidates(self, *, request_id: str, names: Optional[list[str]] = None) -> Dict[str, Any]:
        rid = artifact_safe_name(request_id, fallback="")
        if not rid:
            return {"status": "error", "reason": "request_id_required", "candidates": [], "count": 0}
        run_root = self.run_root_for_request(rid)
        try:
            run_root.relative_to((self.artifact_root / "runs").resolve())
        except ValueError:
            return {"status": "error", "reason": "run_root_outside_artifact_root", "candidates": [], "count": 0}
        output_root = run_root / "outputs"
        manifest = self._read_recovery_manifest(run_root)
        wanted = {artifact_safe_name(item, fallback="") for item in list(names or []) if artifact_safe_name(item, fallback="")}
        candidates: list[Dict[str, Any]] = []
        if output_root.exists():
            for root in sorted(output_root.iterdir(), key=lambda item: item.name):
                if wanted and root.name not in wanted:
                    continue
                files = sorted(path for path in (root.rglob("*") if root.is_dir() else [root]) if path.is_file())
                if not files:
                    continue
                candidates.append(
                    {
                        "name": root.name,
                        "candidate_id": root.name,
                        "path": str(root.resolve()),
                        "file_count": len(files),
                        "size_bytes": sum(int(path.stat().st_size) for path in files),
                        "labels": ["declared_output", "crash_recovery_candidate", "partial_possible"],
                        "files": [
                            {
                                "relative_path": str(path.relative_to(root if root.is_dir() else root.parent)).replace("\\", "/"),
                                "size_bytes": int(path.stat().st_size),
                            }
                            for path in files
                        ],
                    }
                )
        return {
            "status": "ok",
            "contract": "hosting.sandbox.artifact_recovery.v1",
            "request_id": rid,
            "instance_id": str(manifest.get("instance_id") or "").strip() or None,
            "workflow_id": str(manifest.get("workflow_id") or "").strip() or None,
            "package_id": str(manifest.get("package_id") or "").strip() or None,
            "node_id": str(manifest.get("node_id") or "").strip() or None,
            "run_root": str(run_root),
            "crash_or_shutdown_at": time.time(),
            "cleanup_deferred": run_root.exists(),
            "candidates": candidates,
            "count": len(candidates),
        }

    def claim_recovery_artifacts(
        self,
        *,
        request_id: str,
        names: Optional[list[str]] = None,
        target_id: str = "",
        instance_id: str = "",
        patch_absolute_paths: bool = False,
    ) -> Dict[str, Any]:
        inspected = self.recovery_candidates(request_id=request_id, names=names)
        if str(inspected.get("status") or "") != "ok":
            return inspected
        rid = artifact_safe_name(request_id, fallback="request")
        iid = artifact_safe_name(instance_id or inspected.get("instance_id"), fallback="")
        target = artifact_safe_relpath(target_id or (f"instances/{iid}" if iid else f"recovered/{rid}/{int(time.time() * 1000)}"), fallback="recovered")
        run_root = self.run_root_for_request(rid)
        claim_root = (self.roots["artifacts"] / Path(target)).resolve()
        claim_root.mkdir(parents=True, exist_ok=True)
        claimed: list[Dict[str, Any]] = []
        old_to_new_paths: Dict[str, str] = {}
        old_to_new_refs: Dict[str, str] = {}
        for candidate in list(inspected.get("candidates") or []):
            row = dict(candidate or {})
            name = artifact_safe_name(row.get("name"), fallback="artifact")
            source_root = Path(str(row.get("path") or "")).expanduser().resolve()
            try:
                source_root.relative_to(run_root)
            except ValueError:
                continue
            files = sorted(path for path in (source_root.rglob("*") if source_root.is_dir() else [source_root]) if path.is_file())
            for source in files:
                rel = source.relative_to(source_root if source_root.is_dir() else source_root.parent)
                dest = (claim_root / name / rel).resolve()
                try:
                    dest.relative_to(claim_root)
                except ValueError:
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, dest)
                if patch_absolute_paths:
                    self._patch_text_path(dest, old=str(run_root), new=str(claim_root))
                ref_rel = dest.relative_to(self.roots["artifacts"]).as_posix()
                new_ref = f"@artifacts/{ref_rel}"
                old_to_new_paths[str(source)] = str(dest)
                old_to_new_refs[str(source)] = new_ref
                claimed.append(
                    {
                        "name": name,
                        "kind": "ref",
                        "ref": new_ref,
                        "filename": dest.name,
                        "relative_path": str((Path(name) / rel).as_posix()),
                        "size_bytes": int(dest.stat().st_size),
                        "ownership": "host",
                        "labels": list(row.get("labels") or []),
                    }
                )
        return {
            "status": "ok",
            "contract": "hosting.sandbox.artifact_recovery_claim.v1",
            "request_id": rid,
            "instance_id": iid or None,
            "target_id": target,
            "claimed_artifacts": claimed,
            "claimed_count": len(claimed),
            "old_path_to_new_path": old_to_new_paths,
            "old_path_to_new_ref": old_to_new_refs,
        }

    @staticmethod
    def _patch_text_path(path: Path, *, old: str, new: str) -> None:
        if path.suffix.lower() not in {".json", ".txt", ".md", ".csv", ".yaml", ".yml", ".toml"}:
            return
        try:
            if path.stat().st_size > 1024 * 1024:
                return
            text = path.read_text(encoding="utf-8")
            if old in text:
                path.write_text(text.replace(old, new), encoding="utf-8")
        except Exception:
            return


__all__ = [
    "HostedArtifactManager",
    "HostedArtifactRow",
    "artifact_file_output",
    "artifact_has_mask",
    "artifact_host_takeover_output",
    "artifact_inline_input",
    "artifact_inline_zip_input",
    "artifact_inline_zip_output",
    "artifact_masked_ref_input",
    "artifact_path_from_ref",
    "artifact_producer_owned_output",
    "artifact_ref_parts",
    "artifact_ref_input",
    "artifact_safe_name",
    "artifact_safe_relpath",
    "inline_artifact_bytes",
]
