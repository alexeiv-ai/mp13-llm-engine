"""Verified content-addressed storage for signed toolbox wheel bundles."""
from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import shutil
import tempfile
import zipfile
from email.parser import BytesParser
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath
from typing import Any, Mapping
from urllib.parse import parse_qs, urlsplit

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from packaging.markers import default_environment
from packaging.requirements import InvalidRequirement, Requirement
from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.utils import InvalidWheelFilename, parse_wheel_filename
from packaging.version import Version

from ..toolbox.catalog import normalize_distribution_name
from ..toolbox.host_project_config import ToolboxHostProjectConfiguration
from ..toolbox.identity import require_digest
from ..toolbox.target import wheel_is_compatible
from .operation_repository import _exclusive_process_file_lock, _replace_with_bounded_retries


ARTIFACT_STORE_CONTRACT = "hosting.toolbox.artifact_store.v2"
BUNDLE_CONTRACT = "hosting.toolbox.artifact_bundle.v1"
SIGNATURE_CONTRACT = "hosting.toolbox.artifact_bundle_signature.v1"
_BUNDLE_ID_RE = re.compile(r"[a-z0-9]+(?:[._-][a-z0-9]+)*")
_KEY_ID_RE = re.compile(r"[a-z0-9]+(?:[._-][a-z0-9]+)*")
_DIGEST_RE = re.compile(r"sha256:[0-9a-f]{64}")
_MAX_METADATA_BYTES = 1024 * 1024
_MAX_MANIFEST_BYTES = 4 * 1024 * 1024
_MAX_COMPRESSION_RATIO = 100


class _Pep503Parser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.files: list[dict[str, Any]] = []
        self._anchor: dict[str, Any] | None = None

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag.lower() == "a":
            values = {str(key).lower(): str(value or "") for key, value in attrs}
            self._anchor = {
                "href": values.get("href", ""),
                "size": values.get("data-size", ""),
                "text": [],
            }

    def handle_data(self, data: str) -> None:
        if self._anchor is not None:
            self._anchor["text"].append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() != "a" or self._anchor is None:
            return
        anchor = self._anchor
        self._anchor = None
        try:
            digest = parse_qs(urlsplit(anchor["href"]).fragment).get("sha256", [""])[0]
            size = int(anchor["size"])
        except (TypeError, ValueError):
            return
        self.files.append(
            {
                "filename": "".join(anchor["text"]).strip(),
                "hashes": {"sha256": digest},
                "size": size,
            }
        )


class ToolboxArtifactBundleError(RuntimeError):
    """Bounded stable failure; archive paths and parser details are never exposed."""

    _SUMMARIES = {
        "artifact_bundle_archive_invalid": "The artifact bundle ZIP structure is invalid.",
        "artifact_bundle_bounds_exceeded": "The artifact bundle exceeds configured bounds.",
        "artifact_bundle_manifest_invalid": "The artifact bundle manifest is invalid.",
        "artifact_bundle_signature_invalid": "The artifact bundle signature is invalid.",
        "artifact_bundle_target_invalid": "The artifact bundle does not match this host target.",
        "artifact_bundle_artifact_invalid": "An artifact does not match its declared identity.",
        "artifact_bundle_closure_incomplete": "The artifact bundle dependency closure is incomplete.",
        "artifact_bundle_identity_conflict": "The artifact bundle identity conflicts with stored content.",
    }

    def __init__(self, code: str):
        if code not in self._SUMMARIES:
            raise ValueError("artifact_bundle_error_code_invalid")
        self.code = code
        self.summary = self._SUMMARIES[code]
        super().__init__(code)


class ToolboxHttpsArtifactError(RuntimeError):
    _SUMMARIES = {
        "https_metadata_invalid": "The HTTPS package metadata is invalid.",
        "https_metadata_signature_invalid": "The HTTPS package metadata signature is invalid.",
        "https_artifact_target_invalid": "The HTTPS wheel does not match this host target.",
        "https_artifact_invalid": "The HTTPS wheel does not match its signed metadata.",
        "https_artifact_bounds_exceeded": "The HTTPS artifact exceeds configured bounds.",
        "https_artifact_identity_conflict": "The HTTPS artifact conflicts with stored content.",
    }

    def __init__(self, code: str):
        if code not in self._SUMMARIES:
            raise ValueError("https_artifact_error_code_invalid")
        self.code = code
        self.summary = self._SUMMARIES[code]
        super().__init__(code)


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _base64url(value: str, *, length: int) -> bytes:
    text = str(value or "").strip()
    if not text or "=" in text or not re.fullmatch(r"[A-Za-z0-9_-]+", text):
        raise ValueError("base64url_invalid")
    try:
        decoded = base64.urlsafe_b64decode(text + "=" * ((4 - len(text) % 4) % 4))
    except (ValueError, TypeError) as exc:
        raise ValueError("base64url_invalid") from exc
    if len(decoded) != length:
        raise ValueError("base64url_invalid")
    return decoded


def validate_trust_public_keys(
    configuration: ToolboxHostProjectConfiguration,
    trust_public_keys: Mapping[str, str],
) -> dict[str, str]:
    """Require exactly the public keys referenced by configured sources."""
    required = {
        key_id for source in configuration.sources for key_id in source.trust_key_ids
    }
    provided = {str(key): str(value) for key, value in dict(trust_public_keys or {}).items()}
    if set(provided) != required:
        raise ValueError("toolbox_trust_public_keys_invalid")
    try:
        for value in provided.values():
            Ed25519PublicKey.from_public_bytes(_base64url(value, length=32))
    except ValueError as exc:
        raise ValueError("toolbox_trust_public_keys_invalid") from exc
    return provided


class AtomicToolboxArtifactStore:
    def __init__(self, root: Path):
        self.root = Path(root).expanduser().resolve()
        self.index_path = self.root / "index.json"
        self.lock_path = self.root / ".index.lock"
        self.objects_root = self.root / "objects"
        self.staging_root = self.root / ".staging"

    @staticmethod
    def _empty() -> dict[str, Any]:
        return {
            "contract": ARTIFACT_STORE_CONTRACT,
            "bundles": {},
            "https_manifests": {},
            "objects": {},
        }

    @classmethod
    def _validate_index(cls, payload: Mapping[str, Any]) -> dict[str, Any]:
        row = dict(payload or {})
        if set(row) != {
            "contract", "bundles", "https_manifests", "objects"
        } or row.get("contract") != ARTIFACT_STORE_CONTRACT:
            raise ValueError("artifact_store_index_invalid")
        if not isinstance(row["bundles"], dict) or not isinstance(row["objects"], dict):
            raise ValueError("artifact_store_index_invalid")
        for bundle_id, bundle in row["bundles"].items():
            if not _BUNDLE_ID_RE.fullmatch(str(bundle_id)) or not isinstance(bundle, dict):
                raise ValueError("artifact_store_index_invalid")
            if set(bundle) != {
                "manifest_digest", "source_id", "source_set_revision", "target",
                "signing_key_id", "signature", "artifact_digests",
            }:
                raise ValueError("artifact_store_index_invalid")
            require_digest(bundle["manifest_digest"], label="artifact_bundle_manifest_digest")
            require_digest(bundle["source_set_revision"], label="artifact_bundle_source_set_revision")
            if not isinstance(bundle["artifact_digests"], list) or not bundle["artifact_digests"]:
                raise ValueError("artifact_store_index_invalid")
            for digest in bundle["artifact_digests"]:
                require_digest(digest, label="artifact_bundle_artifact_digest")
        for manifest_id, manifest in row["https_manifests"].items():
            if not _BUNDLE_ID_RE.fullmatch(str(manifest_id)) or not isinstance(manifest, dict):
                raise ValueError("artifact_store_index_invalid")
            if set(manifest) != {
                "manifest_digest", "source_id", "source_set_revision", "target",
                "signing_key_id", "signature", "artifact_digests",
            }:
                raise ValueError("artifact_store_index_invalid")
            require_digest(manifest["manifest_digest"], label="https_manifest_digest")
            require_digest(
                manifest["source_set_revision"], label="https_manifest_source_set_revision"
            )
            if not isinstance(manifest["artifact_digests"], list) or not manifest["artifact_digests"]:
                raise ValueError("artifact_store_index_invalid")
            for digest in manifest["artifact_digests"]:
                require_digest(digest, label="https_manifest_artifact_digest")
        for digest, item in row["objects"].items():
            require_digest(digest, label="artifact_store_object_digest")
            if not isinstance(item, dict) or set(item) != {
                "filename", "distribution", "version", "size_bytes", "relative_path"
            }:
                raise ValueError("artifact_store_index_invalid")
            relative = PurePosixPath(str(item["relative_path"]))
            if relative.is_absolute() or ".." in relative.parts or not relative.parts:
                raise ValueError("artifact_store_index_invalid")
        return {
            "contract": ARTIFACT_STORE_CONTRACT,
            "bundles": {str(key): dict(value) for key, value in row["bundles"].items()},
            "https_manifests": {
                str(key): dict(value) for key, value in row["https_manifests"].items()
            },
            "objects": {str(key): dict(value) for key, value in row["objects"].items()},
        }

    def _read_unlocked(self) -> dict[str, Any]:
        if not self.index_path.exists():
            return self._empty()
        try:
            payload = json.loads(self.index_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError("artifact_store_index_corrupt") from exc
        if not isinstance(payload, dict):
            raise ValueError("artifact_store_index_corrupt")
        return self._validate_index(payload)

    def _write_unlocked(self, payload: Mapping[str, Any]) -> None:
        value = self._validate_index(payload)
        self.root.mkdir(parents=True, exist_ok=True)
        descriptor, raw = tempfile.mkstemp(
            prefix=".index.", suffix=".tmp", dir=self.root
        )
        temporary = Path(raw)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
                json.dump(value, handle, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            _replace_with_bounded_retries(temporary, self.index_path)
        finally:
            temporary.unlink(missing_ok=True)

    def read(self) -> dict[str, Any]:
        with _exclusive_process_file_lock(self.lock_path):
            return self._read_unlocked()

    @staticmethod
    def _zip_entry_safe(info: zipfile.ZipInfo) -> bool:
        path = PurePosixPath(info.filename)
        mode = (info.external_attr >> 16) & 0o170000
        return (
            bool(info.filename)
            and "\\" not in info.filename
            and not path.is_absolute()
            and ".." not in path.parts
            and not info.is_dir()
            and mode not in {0o120000, 0o060000, 0o020000}
            and info.compress_type in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}
            and not (info.flag_bits & 0x1)
        )

    @staticmethod
    def _bounded_entry(info: zipfile.ZipInfo, *, maximum_bytes: int) -> bool:
        compressed = max(1, info.compress_size)
        return (
            0 <= info.file_size <= maximum_bytes
            and info.file_size <= compressed * _MAX_COMPRESSION_RATIO
        )

    @staticmethod
    def _strict_json(raw: bytes, *, maximum: int) -> dict[str, Any]:
        if not raw or len(raw) > maximum:
            raise ToolboxArtifactBundleError("artifact_bundle_manifest_invalid")
        try:
            value = json.loads(raw.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise ToolboxArtifactBundleError("artifact_bundle_manifest_invalid") from exc
        if not isinstance(value, dict) or _canonical_bytes(value) != raw:
            raise ToolboxArtifactBundleError("artifact_bundle_manifest_invalid")
        return value

    @staticmethod
    def _verify_wheel_archive(
        path: Path, *, distribution: str, version: str, maximum_bytes: int
    ) -> tuple[Requirement, ...]:
        try:
            with zipfile.ZipFile(path) as wheel:
                entries = wheel.infolist()
                if (
                    not entries
                    or sum(item.file_size for item in entries) > maximum_bytes
                    or any(
                        not AtomicToolboxArtifactStore._zip_entry_safe(item)
                        or not AtomicToolboxArtifactStore._bounded_entry(
                            item, maximum_bytes=maximum_bytes
                        )
                        for item in entries
                    )
                ):
                    raise ToolboxArtifactBundleError("artifact_bundle_artifact_invalid")
                metadata_entries = [
                    item for item in entries if item.filename.endswith(".dist-info/METADATA")
                ]
                if len(metadata_entries) != 1 or metadata_entries[0].file_size > _MAX_METADATA_BYTES:
                    raise ToolboxArtifactBundleError("artifact_bundle_artifact_invalid")
                metadata = BytesParser().parsebytes(wheel.read(metadata_entries[0]))
        except (OSError, zipfile.BadZipFile, RuntimeError) as exc:
            raise ToolboxArtifactBundleError("artifact_bundle_artifact_invalid") from exc
        try:
            if normalize_distribution_name(metadata["Name"]) != distribution or metadata["Version"] != version:
                raise ToolboxArtifactBundleError("artifact_bundle_artifact_invalid")
            requires_python = metadata.get("Requires-Python")
            if requires_python and Version(".".join(str(item) for item in os.sys.version_info[:3])) not in SpecifierSet(requires_python):
                raise ToolboxArtifactBundleError("artifact_bundle_target_invalid")
            return tuple(Requirement(item) for item in metadata.get_all("Requires-Dist", []))
        except (InvalidRequirement, InvalidSpecifier, TypeError, ValueError) as exc:
            if isinstance(exc, ToolboxArtifactBundleError):
                raise
            raise ToolboxArtifactBundleError("artifact_bundle_artifact_invalid") from exc

    def import_signed_bundle(
        self,
        bundle_path: Path,
        *,
        configuration: ToolboxHostProjectConfiguration,
        trust_public_keys: Mapping[str, str],
        expected_source_id: str | None = None,
    ) -> dict[str, Any]:
        if not isinstance(configuration, ToolboxHostProjectConfiguration):
            raise ValueError("toolbox_host_project_configuration_required")
        source = Path(bundle_path).expanduser().resolve()
        maximum_bytes = configuration.resolution.maximum_bytes
        if not source.is_file() or source.stat().st_size > maximum_bytes:
            raise ToolboxArtifactBundleError("artifact_bundle_bounds_exceeded")
        self.staging_root.mkdir(parents=True, exist_ok=True)
        stage = Path(tempfile.mkdtemp(prefix="bundle-", dir=self.staging_root)).resolve()
        try:
            try:
                archive = zipfile.ZipFile(source)
            except (OSError, zipfile.BadZipFile) as exc:
                raise ToolboxArtifactBundleError("artifact_bundle_archive_invalid") from exc
            with archive:
                infos = archive.infolist()
                names = [item.filename for item in infos]
                if (
                    len(infos) < 3
                    or len(infos) > configuration.resolution.maximum_artifacts + 2
                    or len(set(names)) != len(names)
                    or any(not self._zip_entry_safe(item) for item in infos)
                    or any(not self._bounded_entry(item, maximum_bytes=maximum_bytes) for item in infos)
                    or sum(item.file_size for item in infos) > maximum_bytes + 2 * _MAX_MANIFEST_BYTES
                    or "manifest.json" not in names
                    or "signature.json" not in names
                ):
                    raise ToolboxArtifactBundleError("artifact_bundle_archive_invalid")
                manifest_raw = archive.read("manifest.json")
                signature_raw = archive.read("signature.json")
                manifest = self._strict_json(manifest_raw, maximum=_MAX_MANIFEST_BYTES)
                signature = self._strict_json(signature_raw, maximum=_MAX_MANIFEST_BYTES)
                if set(manifest) != {
                    "contract", "bundle_id", "source_id", "source_set_revision", "target",
                    "signing_key_id", "wheels",
                } or manifest.get("contract") != BUNDLE_CONTRACT:
                    raise ToolboxArtifactBundleError("artifact_bundle_manifest_invalid")
                bundle_id = str(manifest.get("bundle_id") or "")
                key_id = str(manifest.get("signing_key_id") or "")
                source_id = str(manifest.get("source_id") or "")
                if expected_source_id is not None and source_id != str(expected_source_id):
                    raise ToolboxArtifactBundleError("artifact_bundle_manifest_invalid")
                if not _BUNDLE_ID_RE.fullmatch(bundle_id) or not _KEY_ID_RE.fullmatch(key_id):
                    raise ToolboxArtifactBundleError("artifact_bundle_manifest_invalid")
                source_config = next(
                    (
                        item for item in configuration.sources
                        if item.source_id == source_id and item.kind == "airgap_store"
                    ),
                    None,
                )
                if (
                    source_config is None
                    or key_id not in source_config.trust_key_ids
                    or manifest.get("source_set_revision") != configuration.source_set_revision
                ):
                    raise ToolboxArtifactBundleError("artifact_bundle_manifest_invalid")
                target = dict(manifest.get("target") or {})
                if target != {
                    "name": configuration.target.name,
                    "python_abi": configuration.target.python_abi,
                    "platform": configuration.target.platform,
                }:
                    raise ToolboxArtifactBundleError("artifact_bundle_target_invalid")
                if set(signature) != {"contract", "algorithm", "key_id", "signature"} or (
                    signature.get("contract") != SIGNATURE_CONTRACT
                    or signature.get("algorithm") != "ed25519"
                    or signature.get("key_id") != key_id
                ):
                    raise ToolboxArtifactBundleError("artifact_bundle_signature_invalid")
                try:
                    public_key = Ed25519PublicKey.from_public_bytes(
                        _base64url(trust_public_keys[key_id], length=32)
                    )
                    public_key.verify(
                        _base64url(signature["signature"], length=64), manifest_raw
                    )
                except (KeyError, ValueError, InvalidSignature) as exc:
                    raise ToolboxArtifactBundleError("artifact_bundle_signature_invalid") from exc
                wheels = manifest.get("wheels")
                if (
                    not isinstance(wheels, list)
                    or not wheels
                    or len(wheels) > configuration.resolution.maximum_artifacts
                ):
                    raise ToolboxArtifactBundleError("artifact_bundle_manifest_invalid")
                expected_names = {"manifest.json", "signature.json"}
                locked_versions: dict[str, str] = {}
                staged: list[dict[str, Any]] = []
                requirements: dict[str, tuple[Requirement, ...]] = {}
                total_wheel_bytes = 0
                for wheel in wheels:
                    if not isinstance(wheel, dict) or set(wheel) != {
                        "distribution", "version", "filename", "size_bytes", "sha256", "tags", "provenance"
                    }:
                        raise ToolboxArtifactBundleError("artifact_bundle_manifest_invalid")
                    distribution = normalize_distribution_name(wheel["distribution"])
                    version = str(wheel["version"] or "")
                    filename = str(wheel["filename"] or "")
                    digest = str(wheel["sha256"] or "")
                    size = wheel["size_bytes"]
                    provenance = str(wheel["provenance"] or "")
                    tags = wheel["tags"]
                    if (
                        distribution in locked_versions
                        or not any(
                            namespace == "*"
                            or distribution == namespace.removesuffix(".*")
                            or (
                                namespace.endswith(".*")
                                and distribution.startswith(namespace[:-1])
                            )
                            for namespace in source_config.allowed_package_namespaces
                        )
                        or not version
                        or not _DIGEST_RE.fullmatch(digest)
                        or isinstance(size, bool)
                        or not isinstance(size, int)
                        or size <= 0
                        or size > maximum_bytes
                        or not provenance
                        or len(provenance.encode("utf-8")) > 512
                        or not isinstance(tags, list)
                        or not tags
                        or tags != sorted(set(tags))
                    ):
                        raise ToolboxArtifactBundleError("artifact_bundle_manifest_invalid")
                    try:
                        wheel_name, wheel_version, _build, wheel_tags = parse_wheel_filename(filename)
                    except InvalidWheelFilename as exc:
                        raise ToolboxArtifactBundleError("artifact_bundle_artifact_invalid") from exc
                    actual_tags = sorted(str(item) for item in wheel_tags)
                    if (
                        normalize_distribution_name(wheel_name) != distribution
                        or str(wheel_version) != version
                        or actual_tags != tags
                        or not wheel_is_compatible(filename, configuration.target)
                    ):
                        raise ToolboxArtifactBundleError("artifact_bundle_target_invalid")
                    entry_name = f"wheels/{filename}"
                    expected_names.add(entry_name)
                    try:
                        info = archive.getinfo(entry_name)
                    except KeyError as exc:
                        raise ToolboxArtifactBundleError("artifact_bundle_artifact_invalid") from exc
                    if info.file_size != size:
                        raise ToolboxArtifactBundleError("artifact_bundle_artifact_invalid")
                    total_wheel_bytes += size
                    if total_wheel_bytes > min(
                        maximum_bytes, source_config.maximum_download_bytes
                    ):
                        raise ToolboxArtifactBundleError("artifact_bundle_bounds_exceeded")
                    staged_path = stage / filename
                    hasher = hashlib.sha256()
                    written = 0
                    with archive.open(info) as source_handle, staged_path.open("wb") as output:
                        while chunk := source_handle.read(1024 * 1024):
                            written += len(chunk)
                            if written > size:
                                raise ToolboxArtifactBundleError("artifact_bundle_artifact_invalid")
                            hasher.update(chunk)
                            output.write(chunk)
                    if written != size or f"sha256:{hasher.hexdigest()}" != digest:
                        raise ToolboxArtifactBundleError("artifact_bundle_artifact_invalid")
                    requirements[distribution] = self._verify_wheel_archive(
                        staged_path,
                        distribution=distribution,
                        version=version,
                        maximum_bytes=maximum_bytes,
                    )
                    locked_versions[distribution] = version
                    staged.append(
                        {
                            "filename": filename,
                            "distribution": distribution,
                            "version": version,
                            "size_bytes": size,
                            "sha256": digest,
                            "staged_path": staged_path,
                        }
                    )
                if set(names) != expected_names:
                    raise ToolboxArtifactBundleError("artifact_bundle_archive_invalid")
                if [item["distribution"] for item in wheels] != sorted(locked_versions):
                    raise ToolboxArtifactBundleError("artifact_bundle_manifest_invalid")
                marker_environment = default_environment()
                for dependencies in requirements.values():
                    for dependency in dependencies:
                        if dependency.marker is not None and not dependency.marker.evaluate(marker_environment):
                            continue
                        dependency_name = normalize_distribution_name(dependency.name)
                        selected = locked_versions.get(dependency_name)
                        if selected is None or (dependency.specifier and Version(selected) not in dependency.specifier):
                            raise ToolboxArtifactBundleError("artifact_bundle_closure_incomplete")

            manifest_digest = "sha256:" + hashlib.sha256(manifest_raw).hexdigest()
            with _exclusive_process_file_lock(self.lock_path):
                index = self._read_unlocked()
                existing = index["bundles"].get(bundle_id)
                bundle_record = {
                    "manifest_digest": manifest_digest,
                    "source_id": source_id,
                    "source_set_revision": configuration.source_set_revision,
                    "target": configuration.target.name,
                    "signing_key_id": key_id,
                    "signature": signature["signature"],
                    "artifact_digests": [item["sha256"] for item in staged],
                }
                if existing is not None:
                    if existing != bundle_record:
                        raise ToolboxArtifactBundleError("artifact_bundle_identity_conflict")
                    return {"status": "already_imported", "bundle_id": bundle_id, **bundle_record}
                for item in staged:
                    digest_hex = item["sha256"].removeprefix("sha256:")
                    relative = PurePosixPath("objects", digest_hex[:2], digest_hex, item["filename"])
                    target_path = self.root.joinpath(*relative.parts)
                    object_record = {
                        "filename": item["filename"],
                        "distribution": item["distribution"],
                        "version": item["version"],
                        "size_bytes": item["size_bytes"],
                        "relative_path": relative.as_posix(),
                    }
                    previous = index["objects"].get(item["sha256"])
                    if previous is not None and previous != object_record:
                        raise ToolboxArtifactBundleError("artifact_bundle_identity_conflict")
                    if not target_path.exists():
                        target_path.parent.mkdir(parents=True, exist_ok=True)
                        os.replace(item["staged_path"], target_path)
                    index["objects"][item["sha256"]] = object_record
                index["bundles"][bundle_id] = bundle_record
                self._write_unlocked(index)
            return {"status": "imported", "bundle_id": bundle_id, **bundle_record}
        except ToolboxArtifactBundleError:
            raise
        except (OSError, ValueError, TypeError, KeyError, zipfile.BadZipFile) as exc:
            raise ToolboxArtifactBundleError("artifact_bundle_manifest_invalid") from exc
        finally:
            shutil.rmtree(stage, ignore_errors=True)

    def import_https_wheel(
        self,
        wheel_path: Path,
        *,
        configuration: ToolboxHostProjectConfiguration,
        source_id: str,
        metadata_raw: bytes,
        signing_key_id: str,
        signature: str,
        trust_public_keys: Mapping[str, str],
        filename: str,
        sha256: str,
        size_bytes: int,
    ) -> dict[str, Any]:
        """Verify one wheel against signed PEP 691 metadata and index it atomically."""
        try:
            source = next(item for item in configuration.sources if item.source_id == source_id)
        except StopIteration as exc:
            raise ToolboxHttpsArtifactError("https_metadata_invalid") from exc
        if source.kind not in {"https_index", "https_artifact"}:
            raise ToolboxHttpsArtifactError("https_metadata_invalid")
        keys = validate_trust_public_keys(configuration, trust_public_keys)
        if signing_key_id not in source.trust_key_ids or signing_key_id not in keys:
            raise ToolboxHttpsArtifactError("https_metadata_signature_invalid")
        try:
            Ed25519PublicKey.from_public_bytes(
                _base64url(keys[signing_key_id], length=32)
            ).verify(_base64url(signature, length=64), bytes(metadata_raw))
        except (InvalidSignature, ValueError) as exc:
            raise ToolboxHttpsArtifactError("https_metadata_signature_invalid") from exc
        if not metadata_raw or len(metadata_raw) > _MAX_MANIFEST_BYTES:
            raise ToolboxHttpsArtifactError("https_metadata_invalid")
        try:
            if metadata_raw.lstrip().startswith(b"{"):
                metadata = json.loads(metadata_raw.decode("utf-8"))
                files = list(metadata["files"])
            else:
                parser = _Pep503Parser()
                parser.feed(metadata_raw.decode("utf-8"))
                parser.close()
                files = parser.files
        except (KeyError, TypeError, ValueError, UnicodeError) as exc:
            raise ToolboxHttpsArtifactError("https_metadata_invalid") from exc
        matches = [item for item in files if isinstance(item, dict) and item.get("filename") == filename]
        if len(matches) != 1:
            raise ToolboxHttpsArtifactError("https_metadata_invalid")
        descriptor = matches[0]
        expected_hex = str(sha256 or "").removeprefix("sha256:")
        if (
            not re.fullmatch(r"[0-9a-f]{64}", expected_hex)
            or descriptor.get("hashes", {}).get("sha256") != expected_hex
            or descriptor.get("size") != size_bytes
            or isinstance(size_bytes, bool)
            or not isinstance(size_bytes, int)
            or size_bytes <= 0
            or size_bytes > min(
                source.maximum_download_bytes, configuration.resolution.maximum_bytes
            )
        ):
            raise ToolboxHttpsArtifactError("https_artifact_bounds_exceeded")
        try:
            wheel_name, wheel_version, _build, _tags = parse_wheel_filename(filename)
        except InvalidWheelFilename as exc:
            raise ToolboxHttpsArtifactError("https_artifact_target_invalid") from exc
        distribution = normalize_distribution_name(wheel_name)
        version = str(wheel_version)
        if not wheel_is_compatible(filename, configuration.target):
            raise ToolboxHttpsArtifactError("https_artifact_target_invalid")
        if not any(
            namespace == "*"
            or distribution == namespace.removesuffix(".*")
            or (namespace.endswith(".*") and distribution.startswith(namespace[:-1]))
            for namespace in source.allowed_package_namespaces
        ):
            raise ToolboxHttpsArtifactError("https_artifact_invalid")
        path = Path(wheel_path).expanduser().resolve()
        try:
            if not path.is_file() or path.name != filename or path.stat().st_size != size_bytes:
                raise ToolboxHttpsArtifactError("https_artifact_invalid")
            hasher = hashlib.sha256()
            with path.open("rb") as handle:
                while chunk := handle.read(1024 * 1024):
                    hasher.update(chunk)
            digest = f"sha256:{hasher.hexdigest()}"
            if digest != f"sha256:{expected_hex}":
                raise ToolboxHttpsArtifactError("https_artifact_invalid")
            self._verify_wheel_archive(
                path,
                distribution=distribution,
                version=version,
                maximum_bytes=min(
                    source.maximum_download_bytes, configuration.resolution.maximum_bytes
                ),
            )
        except ToolboxArtifactBundleError as exc:
            raise ToolboxHttpsArtifactError("https_artifact_invalid") from exc
        except OSError as exc:
            raise ToolboxHttpsArtifactError("https_artifact_invalid") from exc

        manifest_digest = f"sha256:{hashlib.sha256(metadata_raw).hexdigest()}"
        manifest_identity = hashlib.sha256(
            _canonical_bytes(
                {
                    "source_id": source.source_id,
                    "source_set_revision": configuration.source_set_revision,
                    "target": configuration.target.name,
                    "manifest_digest": manifest_digest,
                }
            )
        ).hexdigest()
        manifest_id = f"https-{manifest_identity}"
        manifest_record = {
            "manifest_digest": manifest_digest,
            "source_id": source.source_id,
            "source_set_revision": configuration.source_set_revision,
            "target": configuration.target.name,
            "signing_key_id": signing_key_id,
            "signature": signature,
            "artifact_digests": [digest],
        }
        digest_hex = digest.removeprefix("sha256:")
        relative = PurePosixPath("objects", digest_hex[:2], digest_hex, filename)
        target_path = self.root.joinpath(*relative.parts)
        object_record = {
            "filename": filename,
            "distribution": distribution,
            "version": version,
            "size_bytes": size_bytes,
            "relative_path": relative.as_posix(),
        }
        with _exclusive_process_file_lock(self.lock_path):
            index = self._read_unlocked()
            previous_object = index["objects"].get(digest)
            if previous_object is not None and previous_object != object_record:
                raise ToolboxHttpsArtifactError("https_artifact_identity_conflict")
            previous_manifest = index["https_manifests"].get(manifest_id)
            if previous_manifest is not None:
                immutable = {key: value for key, value in manifest_record.items() if key != "artifact_digests"}
                previous_immutable = {
                    key: value for key, value in previous_manifest.items() if key != "artifact_digests"
                }
                if immutable != previous_immutable:
                    raise ToolboxHttpsArtifactError("https_artifact_identity_conflict")
                manifest_record["artifact_digests"] = sorted(
                    set(previous_manifest["artifact_digests"]) | {digest}
                )
            if not target_path.exists():
                target_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target_path)
            index["objects"][digest] = object_record
            index["https_manifests"][manifest_id] = manifest_record
            self._write_unlocked(index)
        return {"status": "imported", "manifest_id": manifest_id, **manifest_record}

    def object_path(self, digest: str) -> Path:
        key = require_digest(digest, label="artifact_store_object_digest")
        with _exclusive_process_file_lock(self.lock_path):
            item = self._read_unlocked()["objects"].get(key)
        if item is None:
            raise ValueError("artifact_store_object_not_found")
        path = self.root.joinpath(*PurePosixPath(item["relative_path"]).parts).resolve()
        try:
            path.relative_to(self.objects_root.resolve())
        except ValueError as exc:
            raise ValueError("artifact_store_object_path_invalid") from exc
        if not path.is_file():
            raise ValueError("artifact_store_object_missing")
        return path

    def source_artifacts(self, source_id: str) -> dict[str, Path]:
        """Return verified internal object paths for one logical source."""
        logical_source = str(source_id or "").strip()
        with _exclusive_process_file_lock(self.lock_path):
            index = self._read_unlocked()
            digests = sorted(
                {
                    digest
                    for evidence in (
                        *index["bundles"].values(),
                        *index["https_manifests"].values(),
                    )
                    if evidence["source_id"] == logical_source
                    for digest in evidence["artifact_digests"]
                }
            )
            rows = [(digest, dict(index["objects"][digest])) for digest in digests]
        result: dict[str, Path] = {}
        for digest, item in rows:
            path = self.object_path(digest)
            if path.stat().st_size != item["size_bytes"]:
                raise ValueError("artifact_store_object_corrupt")
            hasher = hashlib.sha256()
            with path.open("rb") as handle:
                while chunk := handle.read(1024 * 1024):
                    hasher.update(chunk)
            if f"sha256:{hasher.hexdigest()}" != digest:
                raise ValueError("artifact_store_object_corrupt")
            previous = result.get(item["filename"])
            if previous is not None and previous != path:
                raise ValueError("artifact_store_filename_conflict")
            result[item["filename"]] = path
        return result

    def bundle_evidence_for_artifacts(self, digests: set[str]) -> dict[str, Any]:
        required = {
            require_digest(item, label="artifact_evidence_digest") for item in digests
        }
        if not required:
            raise ValueError("artifact_evidence_required")
        with _exclusive_process_file_lock(self.lock_path):
            index = self._read_unlocked()
            matches = [
                {"bundle_id": bundle_id, **dict(bundle)}
                for bundle_id, bundle in index["bundles"].items()
                if required.issubset(set(bundle["artifact_digests"]))
            ]
        if len(matches) != 1:
            raise ValueError("artifact_evidence_ambiguous")
        return matches[0]

    def verified_evidence_for_artifacts(
        self, digests: set[str], *, source_ids: set[str]
    ) -> dict[str, Any]:
        """Bind a closure to one signed bundle or a deterministic signed HTTPS set."""
        required = {
            require_digest(item, label="artifact_evidence_digest") for item in digests
        }
        expected_sources = {str(item or "").strip() for item in source_ids}
        if not required or not expected_sources or "" in expected_sources:
            raise ValueError("artifact_evidence_required")
        with _exclusive_process_file_lock(self.lock_path):
            index = self._read_unlocked()
            bundles = [
                {"bundle_id": bundle_id, **dict(bundle)}
                for bundle_id, bundle in index["bundles"].items()
                if bundle["source_id"] in expected_sources
                and required.issubset(set(bundle["artifact_digests"]))
            ]
            manifests = {
                manifest_id: dict(manifest)
                for manifest_id, manifest in index["https_manifests"].items()
                if manifest["source_id"] in expected_sources
                and required.intersection(manifest["artifact_digests"])
            }
        if bundles:
            if len(bundles) != 1:
                raise ValueError("artifact_evidence_ambiguous")
            return bundles[0]
        selected: dict[str, dict[str, Any]] = {}
        for digest in sorted(required):
            matches = [
                (manifest_id, manifest)
                for manifest_id, manifest in manifests.items()
                if digest in manifest["artifact_digests"]
            ]
            if len(matches) != 1:
                raise ValueError("artifact_evidence_ambiguous")
            selected[matches[0][0]] = matches[0][1]
        rows = [
            {"manifest_id": manifest_id, **selected[manifest_id]}
            for manifest_id in sorted(selected)
        ]
        if (
            {item["source_set_revision"] for item in rows} != {
                rows[0]["source_set_revision"]
            }
            or {item["target"] for item in rows} != {rows[0]["target"]}
        ):
            raise ValueError("artifact_evidence_ambiguous")
        canonical = _canonical_bytes(
            {
                "source_set_revision": rows[0]["source_set_revision"],
                "target": rows[0]["target"],
                "artifact_digests": sorted(required),
                "manifests": rows,
            }
        )
        evidence_hex = hashlib.sha256(canonical).hexdigest()
        authenticator = base64.urlsafe_b64encode(
            hashlib.sha256(b"".join(item["signature"].encode("ascii") for item in rows)).digest()
        ).decode("ascii").rstrip("=")
        return {
            "evidence_kind": "https_metadata_set",
            "evidence_id": f"https-set-{evidence_hex}",
            "manifest_digest": f"sha256:{evidence_hex}",
            "source_ids": sorted({item["source_id"] for item in rows}),
            "source_set_revision": rows[0]["source_set_revision"],
            "target": rows[0]["target"],
            "signing_key_ids": sorted({item["signing_key_id"] for item in rows}),
            "authenticator": authenticator,
            "artifact_digests": sorted(required),
        }


__all__ = [
    "ARTIFACT_STORE_CONTRACT",
    "AtomicToolboxArtifactStore",
    "BUNDLE_CONTRACT",
    "SIGNATURE_CONTRACT",
    "ToolboxArtifactBundleError",
    "ToolboxHttpsArtifactError",
    "validate_trust_public_keys",
]
