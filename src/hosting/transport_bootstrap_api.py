"""Stable local transport bootstrap artifact/profile API."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from mp13_engine.mp13_config_paths import get_default_config_dir

from .service.host_service import EngineHostService
from .transport_bootstrap import (
    DEFAULT_TRANSPORT_AUTHORIZED_KEY_COMMAND,
    import_transport_bootstrap_bundle,
    install_transport_authorized_key,
    make_transport_bootstrap_bundle,
    provision_client_ssh_artifacts,
    read_transport_bootstrap_bundle,
    validate_client_transport_profile,
    validate_transport_bootstrap_bundle,
    write_transport_bootstrap_bundle,
)


@dataclass(frozen=True)
class TransportBootstrapRequest:
    client_realm_root: Optional[Path] = None
    realm: str = "default"
    profile_name: str = ""
    target: str = ""
    transport_key_id: str = ""
    transport_public_key: str = ""
    transport_private_key_openssh: str = ""
    ssh_known_hosts_line: str = ""
    bundle_file: Optional[Path] = None
    bundle_password: str = ""
    secret_password: str = ""
    overwrite_profile: bool = False
    ssh_alias: str = ""
    overwrite_ssh_config: bool = False
    register_rbac: bool = True
    default_config_dir: Optional[Path] = None
    control_state_file: Optional[Path] = None


def _data(request: TransportBootstrapRequest | Dict[str, Any]) -> Dict[str, Any]:
    return asdict(request) if isinstance(request, TransportBootstrapRequest) else dict(request or {})


def export_transport_bootstrap(request: TransportBootstrapRequest | Dict[str, Any]) -> Dict[str, Any]:
    data = _data(request)
    bundle = make_transport_bootstrap_bundle(
        target=str(data.get("target") or "").strip(),
        ssh_known_hosts_line=str(data.get("ssh_known_hosts_line") or "").strip(),
        transport_key_id=str(data.get("transport_key_id") or "").strip(),
        transport_public_key=str(data.get("transport_public_key") or "").strip(),
        transport_private_key_openssh=str(data.get("transport_private_key_openssh") or "").strip(),
        bundle_password=str(data.get("bundle_password") or ""),
        control_ssh_fingerprint=str(data.get("control_ssh_fingerprint") or "").strip(),
        profile_name=str(data.get("profile_name") or "").strip(),
    )
    result: Dict[str, Any] = {"status": "ok", "bundle": bundle}
    bundle_file = data.get("bundle_file")
    if bundle_file:
        result["bundle_file"] = str(write_transport_bootstrap_bundle(bundle, Path(bundle_file).expanduser().resolve()))
    return result


def import_transport_bootstrap(request: TransportBootstrapRequest | Dict[str, Any]) -> Dict[str, Any]:
    data = _data(request)
    root = Path(data.get("client_realm_root") or "").expanduser().resolve()
    bundle = data.get("bundle")
    if not bundle:
        bundle_file = data.get("bundle_file")
        if not bundle_file:
            raise ValueError("bundle or bundle_file is required")
        bundle = read_transport_bootstrap_bundle(Path(bundle_file).expanduser().resolve())
    return import_transport_bootstrap_bundle(
        bundle=validate_transport_bootstrap_bundle(dict(bundle)),
        client_realm_root=root,
        realm=str(data.get("realm") or "default").strip() or "default",
        profile_name=str(data.get("profile_name") or "").strip() or None,
        overwrite_profile=bool(data.get("overwrite_profile", False)),
        bundle_password=str(data.get("bundle_password") or ""),
        secret_password=str(data.get("secret_password") or ""),
    )


def provision_transport_profile(request: TransportBootstrapRequest | Dict[str, Any]) -> Dict[str, Any]:
    data = _data(request)
    return provision_client_ssh_artifacts(
        client_realm_root=Path(data.get("client_realm_root") or "").expanduser().resolve(),
        profile_name=str(data.get("profile_name") or "").strip(),
        realm=str(data.get("realm") or "default").strip() or "default",
        ssh_alias=str(data.get("ssh_alias") or "").strip(),
        secret_password=str(data.get("secret_password") or ""),
        overwrite=bool(data.get("overwrite_ssh_config", False)),
    )


def validate_transport_profile(request: TransportBootstrapRequest | Dict[str, Any]) -> Dict[str, Any]:
    data = _data(request)
    return validate_client_transport_profile(
        client_realm_root=Path(data.get("client_realm_root") or "").expanduser().resolve(),
        profile_name=str(data.get("profile_name") or "").strip(),
        realm=str(data.get("realm") or "default").strip() or "default",
        run_ssh=bool(data.get("run_ssh", True)),
        ssh_bin=str(data.get("ssh_bin") or "ssh").strip() or "ssh",
        remote_command=str(data.get("remote_command") or "exit 0").strip() or "exit 0",
        timeout_seconds=float(data.get("timeout_seconds", 15.0) or 15.0),
        secret_password=str(data.get("secret_password") or ""),
    )


def install_authorized_transport_key(request: Dict[str, Any]) -> Dict[str, Any]:
    data = dict(request or {})
    authorized_keys_file = data.get("authorized_keys_file")
    if not authorized_keys_file:
        raise ValueError("authorized_keys_file is required")
    transport_public_key = str(data.get("transport_public_key") or "").strip()
    install_result = install_transport_authorized_key(
        transport_public_key=transport_public_key,
        authorized_keys_file=Path(authorized_keys_file).expanduser().resolve(),
        transport_key_id=str(data.get("transport_key_id") or "").strip(),
        marker=str(data.get("marker") or "mp13-hosting-transport").strip(),
        forced_command=str(data.get("forced_command") or DEFAULT_TRANSPORT_AUTHORIZED_KEY_COMMAND).strip(),
        restrict_options=bool(data.get("restrict_options", True)),
    )
    if not bool(data.get("register_rbac", True)):
        return install_result
    control_state_value = data.get("control_state_file")
    if control_state_value:
        control_state_file = Path(control_state_value).expanduser().resolve()
    else:
        default_config_value = data.get("default_config_dir")
        default_config_dir = (
            Path(default_config_value).expanduser().resolve()
            if default_config_value
            else get_default_config_dir()
        )
        control_state_file = (default_config_dir / "hosting" / "access_control.json").resolve()
    key_result = EngineHostService(control_state_file=control_state_file).auth_upsert_key(
        key_id=str(install_result.get("transport_key_id") or "transport"),
        role="transport",
        auth_method="public_key",
        public_key=transport_public_key,
    )
    return {
        **install_result,
        "control_state_file": str(control_state_file),
        "rbac_key_id": key_result.get("key_id"),
        "rbac_role": key_result.get("role"),
        "rbac_auth_method": key_result.get("auth_method"),
    }


def install_authorized_transport_key_file_only(request: Dict[str, Any]) -> Dict[str, Any]:
    data = dict(request or {})
    data["register_rbac"] = False
    authorized_keys_file = data.get("authorized_keys_file")
    if not authorized_keys_file:
        raise ValueError("authorized_keys_file is required")
    return install_transport_authorized_key(
        transport_public_key=str(data.get("transport_public_key") or "").strip(),
        authorized_keys_file=Path(authorized_keys_file).expanduser().resolve(),
        transport_key_id=str(data.get("transport_key_id") or "").strip(),
        marker=str(data.get("marker") or "mp13-hosting-transport").strip(),
        forced_command=str(data.get("forced_command") or DEFAULT_TRANSPORT_AUTHORIZED_KEY_COMMAND).strip(),
        restrict_options=bool(data.get("restrict_options", True)),
    )


__all__ = [
    "TransportBootstrapRequest",
    "export_transport_bootstrap",
    "import_transport_bootstrap",
    "provision_transport_profile",
    "validate_transport_profile",
    "install_authorized_transport_key",
    "install_authorized_transport_key_file_only",
]
