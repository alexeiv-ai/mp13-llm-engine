from __future__ import annotations

import hashlib
import json
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urljoin, urlsplit, urlunsplit

import requests
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from packaging.utils import InvalidWheelFilename, parse_wheel_filename

from ..toolbox.catalog import normalize_distribution_name
from ..toolbox.host_project_config import ToolboxHostProjectConfiguration, ToolboxPackageSource
from ..toolbox.target import wheel_is_compatible
from .toolbox_artifact_store import (
    AtomicToolboxArtifactStore,
    ToolboxHttpsArtifactError,
    _base64url,
    validate_trust_public_keys,
)


_SIGNATURE_HEADER = "X-MP13-Signature"
_KEY_ID_HEADER = "X-MP13-Signing-Key-Id"
_MAX_REDIRECTS = 5
_MAX_METADATA_BYTES = 4 * 1024 * 1024


class ToolboxHttpsAcquisitionError(RuntimeError):
    _SUMMARIES = {
        "https_source_credentials_invalid": "The HTTPS source credential bindings are invalid.",
        "https_source_request_failed": "The HTTPS package source request failed.",
        "https_source_redirect_denied": "The HTTPS package source redirect is not allowed.",
        "https_source_metadata_invalid": "The HTTPS package source metadata is invalid.",
        "https_source_signature_invalid": "The HTTPS package source signature is invalid.",
        "https_source_artifact_invalid": "The HTTPS wheel does not match signed source metadata.",
        "https_source_bounds_exceeded": "The HTTPS source exceeded configured byte bounds.",
    }

    def __init__(self, code: str):
        if code not in self._SUMMARIES:
            raise ValueError("https_acquisition_error_code_invalid")
        self.code = code
        self.summary = self._SUMMARIES[code]
        super().__init__(code)


class ToolboxHttpsArtifactAcquirer:
    """Fetch signed PEP 691 metadata and exact wheels into the shared CAS."""

    def __init__(
        self,
        configuration: ToolboxHostProjectConfiguration,
        *,
        artifact_store: AtomicToolboxArtifactStore,
        trust_public_keys: Mapping[str, str],
        source_credentials: Mapping[str, str] | None = None,
        session: Any = None,
    ) -> None:
        self.configuration = configuration
        self.artifact_store = artifact_store
        self.trust_public_keys = validate_trust_public_keys(
            configuration, trust_public_keys
        )
        required_credentials = {
            source.credential_ref
            for source in configuration.sources
            if source.kind in {"https_index", "https_artifact"}
            and source.credential_ref is not None
        }
        provided = {
            str(key): str(value)
            for key, value in dict(source_credentials or {}).items()
        }
        if set(provided) != required_credentials or any(
            not value
            or len(value.encode("utf-8")) > 4096
            or "\r" in value
            or "\n" in value
            for value in provided.values()
        ):
            raise ToolboxHttpsAcquisitionError("https_source_credentials_invalid")
        self.source_credentials = provided
        self.session = session or requests.Session()

    def _source(self, source_id: str) -> ToolboxPackageSource:
        try:
            source = next(
                item for item in self.configuration.sources if item.source_id == source_id
            )
        except StopIteration as exc:
            raise ToolboxHttpsAcquisitionError("https_source_metadata_invalid") from exc
        if source.kind not in {"https_index", "https_artifact"}:
            raise ToolboxHttpsAcquisitionError("https_source_metadata_invalid")
        return source

    @staticmethod
    def _origin(url: str) -> str:
        parsed = urlsplit(url)
        if parsed.scheme.lower() != "https" or not parsed.hostname or parsed.username or parsed.password:
            raise ToolboxHttpsAcquisitionError("https_source_redirect_denied")
        port = f":{parsed.port}" if parsed.port is not None else ""
        return f"https://{parsed.hostname.lower()}{port}"

    def _allowed_origins(self, source: ToolboxPackageSource) -> set[str]:
        return {
            self._origin(source.origin),
            *(self._origin(item) for item in self.configuration.resolution.allowed_redirect_origins),
        }

    def _headers(self, source: ToolboxPackageSource, *, metadata: bool) -> dict[str, str]:
        headers = {
            "Accept": (
                "application/vnd.pypi.simple.v1+json, application/json"
                if metadata
                else "application/octet-stream"
            ),
            "User-Agent": "mp13-toolbox-host/1",
        }
        if source.credential_ref is not None:
            headers["Authorization"] = self.source_credentials[source.credential_ref]
        return headers

    def _request(self, source: ToolboxPackageSource, url: str, *, metadata: bool):
        current = str(url)
        allowed = self._allowed_origins(source)
        for _redirect in range(_MAX_REDIRECTS + 1):
            if self._origin(current) not in allowed:
                raise ToolboxHttpsAcquisitionError("https_source_redirect_denied")
            try:
                response = self.session.get(
                    current,
                    headers=self._headers(source, metadata=metadata),
                    timeout=self.configuration.resolution.timeout_seconds,
                    allow_redirects=False,
                    stream=True,
                )
            except requests.RequestException as exc:
                raise ToolboxHttpsAcquisitionError("https_source_request_failed") from exc
            if response.status_code in {301, 302, 303, 307, 308}:
                location = response.headers.get("Location")
                response.close()
                if not location:
                    raise ToolboxHttpsAcquisitionError("https_source_redirect_denied")
                current = urljoin(current, location)
                continue
            if response.status_code != 200:
                response.close()
                raise ToolboxHttpsAcquisitionError("https_source_request_failed")
            return response, current
        raise ToolboxHttpsAcquisitionError("https_source_redirect_denied")

    @staticmethod
    def _read_bounded(response: Any, maximum_bytes: int) -> bytes:
        content_length = response.headers.get("Content-Length")
        if content_length is not None:
            try:
                if int(content_length) < 0 or int(content_length) > maximum_bytes:
                    raise ToolboxHttpsAcquisitionError("https_source_bounds_exceeded")
            except ValueError as exc:
                raise ToolboxHttpsAcquisitionError("https_source_bounds_exceeded") from exc
        chunks: list[bytes] = []
        total = 0
        try:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                total += len(chunk)
                if total > maximum_bytes:
                    raise ToolboxHttpsAcquisitionError("https_source_bounds_exceeded")
                chunks.append(bytes(chunk))
        finally:
            response.close()
        return b"".join(chunks)

    def _verify_metadata_signature(
        self, source: ToolboxPackageSource, raw: bytes, headers: Mapping[str, Any]
    ) -> tuple[str, str]:
        key_id = str(headers.get(_KEY_ID_HEADER) or "").strip()
        signature = str(headers.get(_SIGNATURE_HEADER) or "").strip()
        if key_id not in source.trust_key_ids:
            raise ToolboxHttpsAcquisitionError("https_source_signature_invalid")
        try:
            key = Ed25519PublicKey.from_public_bytes(
                _base64url(self.trust_public_keys[key_id], length=32)
            )
            key.verify(_base64url(signature, length=64), raw)
        except (KeyError, InvalidSignature, ValueError) as exc:
            raise ToolboxHttpsAcquisitionError("https_source_signature_invalid") from exc
        return key_id, signature

    def fetch_project_metadata(self, *, source_id: str, project_name: str) -> dict[str, Any]:
        source = self._source(source_id)
        normalized = normalize_distribution_name(project_name)
        url = (
            f"{source.origin.rstrip('/')}/{normalized}/"
            if source.kind == "https_index"
            else source.origin
        )
        response, final_url = self._request(source, url, metadata=True)
        headers = dict(response.headers)
        raw = self._read_bounded(
            response,
            min(_MAX_METADATA_BYTES, source.maximum_download_bytes),
        )
        key_id, signature = self._verify_metadata_signature(source, raw, headers)
        try:
            payload = json.loads(raw.decode("utf-8"))
            files = list(payload["files"])
        except (KeyError, TypeError, ValueError, UnicodeError) as exc:
            raise ToolboxHttpsAcquisitionError("https_source_metadata_invalid") from exc
        if payload.get("name") is not None and normalize_distribution_name(payload["name"]) != normalized:
            raise ToolboxHttpsAcquisitionError("https_source_metadata_invalid")
        eligible: list[dict[str, Any]] = []
        for item in files:
            if not isinstance(item, dict):
                raise ToolboxHttpsAcquisitionError("https_source_metadata_invalid")
            filename = str(item.get("filename") or "")
            try:
                wheel_name, wheel_version, _build, _tags = parse_wheel_filename(filename)
            except InvalidWheelFilename:
                continue
            if (
                normalize_distribution_name(wheel_name) != normalized
                or not wheel_is_compatible(filename, self.configuration.target)
            ):
                continue
            digest_hex = str(dict(item.get("hashes") or {}).get("sha256") or "")
            size = item.get("size")
            artifact_url = urljoin(final_url, str(item.get("url") or ""))
            if (
                not re.fullmatch(r"[0-9a-f]{64}", digest_hex)
                or isinstance(size, bool)
                or not isinstance(size, int)
                or size <= 0
                or size > source.maximum_download_bytes
                or self._origin(artifact_url) not in self._allowed_origins(source)
            ):
                raise ToolboxHttpsAcquisitionError("https_source_metadata_invalid")
            eligible.append(
                {
                    "filename": filename,
                    "distribution": normalized,
                    "version": str(wheel_version),
                    "url": urlunsplit(urlsplit(artifact_url)._replace(fragment="")),
                    "sha256": f"sha256:{digest_hex}",
                    "size_bytes": size,
                }
            )
        if not eligible or len(eligible) > self.configuration.resolution.maximum_artifacts:
            raise ToolboxHttpsAcquisitionError("https_source_metadata_invalid")
        eligible.sort(key=lambda item: (item["version"], item["filename"]))
        return {
            "source_id": source.source_id,
            "project": normalized,
            "metadata_raw": raw,
            "metadata_digest": f"sha256:{hashlib.sha256(raw).hexdigest()}",
            "signing_key_id": key_id,
            "signature": signature,
            "files": eligible,
        }

    def acquire_wheel(
        self, *, metadata: Mapping[str, Any], artifact: Mapping[str, Any]
    ) -> dict[str, Any]:
        source = self._source(str(metadata.get("source_id") or ""))
        item = dict(artifact or {})
        filename = str(item.get("filename") or "")
        if item not in list(metadata.get("files") or []):
            raise ToolboxHttpsAcquisitionError("https_source_artifact_invalid")
        response, _final_url = self._request(source, str(item.get("url") or ""), metadata=False)
        maximum = min(
            int(item["size_bytes"]),
            source.maximum_download_bytes,
            self.configuration.resolution.maximum_bytes,
        )
        raw = self._read_bounded(response, maximum)
        if len(raw) != item["size_bytes"] or (
            f"sha256:{hashlib.sha256(raw).hexdigest()}" != item["sha256"]
        ):
            raise ToolboxHttpsAcquisitionError("https_source_artifact_invalid")
        self.artifact_store.staging_root.mkdir(parents=True, exist_ok=True)
        stage = Path(tempfile.mkdtemp(prefix="https-", dir=self.artifact_store.staging_root))
        wheel_path = stage / filename
        try:
            wheel_path.write_bytes(raw)
            imported = self.artifact_store.import_https_wheel(
                wheel_path,
                configuration=self.configuration,
                source_id=source.source_id,
                metadata_raw=bytes(metadata["metadata_raw"]),
                signing_key_id=str(metadata["signing_key_id"]),
                signature=str(metadata["signature"]),
                trust_public_keys=self.trust_public_keys,
                filename=filename,
                sha256=str(item["sha256"]),
                size_bytes=int(item["size_bytes"]),
            )
        except ToolboxHttpsArtifactError as exc:
            raise ToolboxHttpsAcquisitionError("https_source_artifact_invalid") from exc
        finally:
            shutil.rmtree(stage, ignore_errors=True)
        return {
            **imported,
            "filename": filename,
            "sha256": item["sha256"],
            "size_bytes": item["size_bytes"],
        }


__all__ = ["ToolboxHttpsAcquisitionError", "ToolboxHttpsArtifactAcquirer"]
