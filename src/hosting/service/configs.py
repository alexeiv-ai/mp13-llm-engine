"""Engine config-store helpers for the engine host service."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from .constants import ROLE_MODEL_USER, ROLE_MODEL_USER_WITH_MODEL_CONTROL


class ConfigMixin:
    @staticmethod
    def _connectivity_mode(cfg: Dict[str, Any]) -> str:
        access_profile = dict(cfg.get("access_profile") or {})
        raw = str(access_profile.get("connectivity_mode") or "local_only").strip().lower()
        if raw in {"local_only", "ssh_tunnel_only", "truly_remote"}:
            return raw
        return "local_only"

    def _requires_ssh_binding(self, cfg: Dict[str, Any]) -> bool:
        return self._connectivity_mode(cfg) != "local_only"

    def _classify_connect_worker_class(
        self,
        *,
        config_path: str,
        payload: Optional[Dict[str, Any]],
    ) -> str:
        p = dict(payload or {})
        cfg: Dict[str, Any] = {}
        try:
            cfg = self._merge_default_and_selected_config(config_path)
        except Exception:
            cfg = {}

        def _norm(v: Any) -> str:
            return str(v or "").strip().lower()

        hosting_cfg = dict(cfg.get("hosting") or {}) if isinstance(cfg.get("hosting"), dict) else {}
        marker = _norm(
            cfg.get("worker_kind")
            or cfg.get("worker_type")
            or hosting_cfg.get("worker_kind")
            or hosting_cfg.get("worker_type")
        )
        if marker in {"generic", "non_model", "worker", "generic_worker"}:
            return "generic"
        if marker in {"model", "model_engine", "engine"}:
            return "model"

        worker_command = cfg.get("worker_command")
        spawn_cfg = dict(cfg.get("spawn") or {}) if isinstance(cfg.get("spawn"), dict) else {}
        spawn_command = spawn_cfg.get("command")
        if isinstance(worker_command, list) and worker_command:
            return "generic"
        if isinstance(spawn_command, list) and spawn_command:
            return "generic"

        configured_model = (
            ((cfg.get("engine_params") or {}).get("base_model_path") if isinstance(cfg.get("engine_params"), dict) else None)
            or cfg.get("base_model_path")
            or cfg.get("model")
            or cfg.get("base_model_name_or_path")
        )
        if str(p.get("model_path") or "").strip() or str(configured_model or "").strip():
            return "model"
        return "unknown"

    @staticmethod
    def _normalize_worker_profile_class(value: Optional[str]) -> str:
        v = str(value or "").strip().lower()
        if v in {"model", "generic"}:
            return v
        return "unknown"

    def _authorize_role_for_engine_profile(self, *, role: str, engine_id: str) -> None:
        eid = str(engine_id or "").strip()
        if not eid:
            return
        reg = self._find_registration(eid)
        if not isinstance(reg, dict):
            return
        profile = self._normalize_worker_profile_class(str(reg.get("worker_profile_class") or ""))
        executor_kind = str(reg.get("executor_kind") or "").strip().lower()
        r = str(role or "").strip().lower()
        if (
            profile == "generic"
            and executor_kind != "workflow_js_helper"
            and r in {ROLE_MODEL_USER, ROLE_MODEL_USER_WITH_MODEL_CONTROL}
        ):
            raise PermissionError("insufficient_role")

    @staticmethod
    def _safe_config_name(value: str) -> str:
        cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", (value or "").strip()).strip("_")
        return cleaned or "engine_config"

    def _logs_dir(self) -> Path:
        return self.engines_state_file.parent / "logs"

    def _engine_log_path(self, engine_id: str) -> Path:
        stem = self._safe_config_name(str(engine_id or "engine"))
        return (self._logs_dir() / f"{stem}.log").expanduser().resolve()

    def _default_config_path(self) -> Path:
        try:
            from mp13_engine.mp13_config_paths import get_default_config_dir  # type: ignore

            p = Path(get_default_config_dir()) / "mp13_config.json"
            return p.expanduser().resolve()
        except Exception:
            return (Path.home() / ".mp13-llm" / "mp13_config.json").expanduser().resolve()

    def _config_store_dir(self) -> Path:
        base = self._default_config_path().parent
        return (base / "backend" / "configs").expanduser().resolve()

    def _config_store_mode(self) -> str:
        control = self._read_control()
        cfg = dict(control.get("control_config") or {})
        mode = str(cfg.get("config_store_mode") or "store_only").strip().lower()
        return mode if mode in {"store_only"} else "store_only"

    @staticmethod
    def _normalize_traffic_policy(policy: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        p = dict(policy or {})
        allowed_methods = [str(x or "").strip().upper() for x in list(p.get("allowed_methods") or []) if str(x or "").strip()]
        if not allowed_methods:
            allowed_methods = ["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"]
        allowed_path_prefixes = [str(x or "").strip() for x in list(p.get("allowed_path_prefixes") or []) if str(x or "").strip()]
        if not allowed_path_prefixes:
            allowed_path_prefixes = ["/"]
        req_headers = [str(x or "").strip().lower() for x in list(p.get("request_header_allowlist") or []) if str(x or "").strip()]
        resp_headers = [str(x or "").strip().lower() for x in list(p.get("response_header_allowlist") or []) if str(x or "").strip()]
        if not req_headers:
            req_headers = ["accept", "content-type", "authorization", "x-request-id", "x-trace-id", "x-correlation-id", "user-agent"]
        if not resp_headers:
            resp_headers = ["content-type", "content-length", "cache-control", "etag", "last-modified", "x-request-id", "x-trace-id", "x-correlation-id", "date", "server"]
        max_req = max(1024, int(p.get("max_request_bytes") or (1024 * 1024)))
        max_resp = max(1024, int(p.get("max_response_bytes") or (1024 * 1024)))
        return {
            "allowed_methods": sorted(list(set(allowed_methods))),
            "allowed_path_prefixes": sorted(list(set(allowed_path_prefixes))),
            "request_header_allowlist": sorted(list(set(req_headers))),
            "response_header_allowlist": sorted(list(set(resp_headers))),
            "allow_authorization_header": bool(p.get("allow_authorization_header", False)),
            "max_request_bytes": max_req,
            "max_response_bytes": max_resp,
        }

    def _traffic_policy(self) -> Dict[str, Any]:
        control = self._read_control()
        cfg = dict(control.get("control_config") or {})
        return self._normalize_traffic_policy(dict(cfg.get("traffic_policy") or {}))

    def _traffic_policy_for_engine(self, engine_id: str) -> Dict[str, Any]:
        control = self._read_control()
        cfg = dict(control.get("control_config") or {})
        base = self._normalize_traffic_policy(dict(cfg.get("traffic_policy") or {}))
        engine_policies = dict(cfg.get("engine_traffic_policies") or {})
        eid = self._safe_config_name(str(engine_id or "").strip())
        override = dict(engine_policies.get(eid) or {})
        if not override:
            return base
        merged = dict(base)
        merged.update(override)
        return self._normalize_traffic_policy(merged)

    def _normalize_config_selector(self, config_path: str) -> str:
        raw = str(config_path or "").strip()
        if not raw or raw.lower() == "default":
            return "default"
        if any(x in raw for x in ["/", "\\", ":"]) or raw.startswith("."):
            raise ValueError("config_path must be 'default' or a config name in hosted config store")
        stem = Path(raw if Path(raw).suffix else f"{raw}.json").stem
        safe = self._safe_config_name(stem)
        if safe != stem:
            raise ValueError("config_path contains unsupported characters")
        return safe

    def _resolve_json_config_path(self, config_path: str) -> Path:
        default_path = self._default_config_path()
        selector = self._normalize_config_selector(config_path)
        if selector == "default":
            return default_path
        if self._config_store_mode() != "store_only":
            raise ValueError("Unsupported config store mode")
        return (self._config_store_dir() / f"{selector}.json").expanduser().resolve()

    def _merge_default_and_selected_config(self, config_path: str) -> Dict[str, Any]:
        default_path = self._default_config_path()
        selected_path = self._resolve_json_config_path(config_path)
        default_data: Dict[str, Any] = {}
        selected_data: Dict[str, Any] = {}
        if default_path.exists():
            try:
                default_data = json.loads(default_path.read_text(encoding="utf-8")) or {}
            except Exception:
                default_data = {}
        if selected_path.exists():
            selected_data = json.loads(selected_path.read_text(encoding="utf-8")) or {}
        if selected_path.resolve() == default_path.resolve():
            return selected_data if isinstance(selected_data, dict) else {}
        merged = dict(default_data) if isinstance(default_data, dict) else {}
        if isinstance(selected_data, dict):
            for k, v in selected_data.items():
                if isinstance(v, dict) and isinstance(merged.get(k), dict):
                    nested = dict(merged[k])
                    nested.update(v)
                    merged[k] = nested
                else:
                    merged[k] = v
        return merged

    @staticmethod
    def _resolve_path_token(value: str, *, config_dir: Path) -> Path:
        from mp13_engine.mp13_config_paths import PathResolver, detect_project_root

        raw = str(value or "").strip()
        if not raw:
            return config_dir
        cwd = Path.cwd().resolve()
        resolver = PathResolver(
            cwd=cwd,
            config_dir=config_dir.resolve(),
            home_dir=Path.home().resolve(),
            project_dir=detect_project_root(cwd),
            category_roots={},
        )
        return Path(str(resolver.resolve(raw))).resolve()

    def list_engine_configs(self) -> List[Dict[str, Any]]:
        default_path = self._default_config_path()
        out: List[Dict[str, Any]] = []
        seen: set[str] = set()
        engine_python = self._engine_python_executable()
        engine_runtime_ok, _engine_runtime_err = self._check_module_discoverable(engine_python, "mp13_engine")

        def _config_meta(selector: str) -> Dict[str, Any]:
            try:
                _ = self._merge_default_and_selected_config(selector)
            except Exception as e:
                return {"has_spawn_command": False, "connect_reason": f"invalid_config: {e}"}
            return {
                "has_spawn_command": bool(engine_runtime_ok),
                "connect_reason": None if engine_runtime_ok else "engine_not_available",
            }

        if default_path.exists():
            row = {"name": "default", "path": str(default_path), "is_default": True}
            row.update(_config_meta("default"))
            out.append(row)
            seen.add(str(default_path.resolve()))
        cfg_dir = self._config_store_dir()
        if cfg_dir.exists():
            for fp in sorted(cfg_dir.glob("*.json"), key=lambda p: p.name.lower()):
                try:
                    rp = str(fp.resolve())
                except Exception:
                    rp = str(fp)
                if rp in seen:
                    continue
                row = {"name": fp.stem, "path": str(fp), "is_default": False}
                row.update(_config_meta(fp.stem))
                out.append(row)
                seen.add(rp)
        return out

    def create_engine_config(self, *, name: str, config: Dict[str, Any], overwrite: bool = False) -> Dict[str, Any]:
        cfg_dir = self._config_store_dir()
        cfg_dir.mkdir(parents=True, exist_ok=True)
        stem = self._safe_config_name(name)
        path = (cfg_dir / f"{stem}.json").resolve()
        if path.exists() and not bool(overwrite):
            raise ValueError(f"Config '{stem}' already exists")
        existed = path.exists()
        payload = dict(config or {})
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return {"name": stem, "path": str(path), "created": True, "overwrote": bool(existed and overwrite)}

    def models_from_config(self, config_path: str) -> List[Dict[str, Any]]:
        from mp13_engine.mp13_config_paths import resolve_config_paths

        cfg = self._merge_default_and_selected_config(config_path)
        selected_path = self._resolve_json_config_path(config_path)
        _resolved_cfg, resolver = resolve_config_paths(
            cfg,
            cwd=selected_path.parent,
            config_path=selected_path,
            project_root=self._service_project_root(),
        )
        models_root = resolver.category_roots.get("models") or selected_path.parent
        results: List[Dict[str, Any]] = []
        if not models_root.exists():
            return results
        for child in models_root.iterdir():
            if not child.is_dir():
                continue
            safes = list(child.glob("*.safetensors"))
            if safes:
                results.append({"name": child.name, "path": str(child), "safetensors_count": len(safes)})
        results.sort(key=lambda x: str(x.get("name") or "").lower())
        return results

    def _resolve_model_path_from_config_value(self, value: str, *, config_path: str, cfg: Dict[str, Any]) -> str:
        from mp13_engine.mp13_config_paths import resolve_config_paths

        raw = str(value or "").strip()
        if not raw:
            return ""
        selected_path = self._resolve_json_config_path(config_path)
        _resolved_cfg, resolver = resolve_config_paths(
            cfg,
            cwd=selected_path.parent,
            config_path=selected_path,
            project_root=self._service_project_root(),
        )
        resolved = resolver.resolve(raw, category="models", allow_remote_id=False)
        return str(resolved or raw)

    @staticmethod
    def _service_project_root() -> Optional[Path]:
        from mp13_engine.mp13_config_paths import detect_project_root

        return detect_project_root(Path(__file__).resolve())
