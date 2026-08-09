"""Toolbox runtime routing, execution, and registration orchestration."""
from __future__ import annotations

import json
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from mp13_engine.mp13_toolbox import ToolsView

from ..callable_surface import HOST_CAPABILITY_APPROVAL_CALLBACK_NAME, HOST_CAPABILITY_DISPATCH_CALLBACK_NAME, host_capability_approval_request
from ..operation_contract import (
    HostedExecutionKind,
    HostedOperationLifecycle,
    HostedOperationSelector,
    hosted_execution_fingerprint,
)
from ..sandbox.host_capabilities import HostCapabilityBroker
from ..sandbox.service_broker_registry import service_broker_host_capability_session
from ..toolbox.callbacks import _HostedToolCallbackRelay
from .errors import ToolboxRolloutError


class ToolboxRuntimeMixin:
    def _apply_resolved_toolbox_definition(
        self,
        *,
        draft: Any,
        profile_changes: List[Dict[str, Any]],
        operation_id: str,
    ) -> Dict[str, Any]:
        from .toolbox_rollout import ToolboxDefinitionRolloutCoordinator

        return self._run_locked_toolbox_call(
            str(draft.definition.toolbox_id),
            ToolboxDefinitionRolloutCoordinator(self).apply,
            draft=draft,
            profile_changes=profile_changes,
            operation_id=str(operation_id or "").strip(),
        )

    def _cleanup_toolbox_definition_apply_candidates(self, *, record: Dict[str, Any]) -> Dict[str, Any]:
        metadata = dict(dict(record or {}).get("metadata") or {})
        candidates = sorted(
            {
                str(item or "").strip()
                for item in list(metadata.get("candidate_engine_ids") or [])
                if str(item or "").strip()
            }
        )
        toolbox_id = str(metadata.get("toolbox_id") or "").strip()
        active = self._active_toolbox_v2_snapshot(toolbox_id) if toolbox_id else None
        active_engine_ids = {
            str(dict(route or {}).get("engine_id") or "").strip()
            for route in dict(dict(active or {}).get("tool_routes") or {}).values()
        }
        cleaned: list[str] = []
        for engine_id in candidates:
            if engine_id in active_engine_ids:
                continue
            self._retire_toolbox_registration(engine_id)
            cleaned.append(engine_id)
        return {"status": "complete", "candidate_count": len(cleaned)}

    def toolbox_definition_apply_operator_details(
        self,
        *,
        operation_id: str,
        operator_authorized: bool,
    ) -> Dict[str, Any]:
        if not bool(operator_authorized):
            raise PermissionError("toolbox_operator_details_denied")
        oid = str(operation_id or "").strip()
        if not oid or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._:-" for character in oid):
            raise ValueError("operation_id_invalid")
        path = (
            self.hosting_root / "state" / "toolbox_rollout_operator_details" / f"{oid}.json"
        ).resolve()
        try:
            path.relative_to((self.hosting_root / "state" / "toolbox_rollout_operator_details").resolve())
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
            raise ValueError("toolbox_operator_details_unavailable") from exc
        if not isinstance(payload, dict):
            raise ValueError("toolbox_operator_details_unavailable")
        return payload

    @staticmethod
    def _toolbox_operation_namespace(*, engine_id: str = "", toolbox_id: str = "") -> str:
        tid = str(toolbox_id or "").strip()
        eid = str(engine_id or "").strip()
        if tid:
            return f"toolbox:{tid}"
        if eid:
            return f"engine:{eid}"
        raise ValueError("engine_id or toolbox_id is required")

    @staticmethod
    def _registration_allowed_tool_names(reg: Dict[str, Any]) -> Optional[set[str]]:
        tool_access = dict(reg.get("tool_access") or {})
        allowed = {
            str(item or "").strip()
            for item in list(tool_access.get("allowed_tool_names") or [])
            if str(item or "").strip()
        }
        return allowed or None

    @staticmethod
    def _registration_advertised_tool_names(reg: Dict[str, Any]) -> Optional[set[str]]:
        tool_access = dict(reg.get("tool_access") or {})
        advertised = {
            str(item or "").strip()
            for item in list(tool_access.get("advertised_tool_names") or [])
            if str(item or "").strip()
        }
        return advertised or None

    @staticmethod
    def _registration_hidden_allowed_tool_names(reg: Dict[str, Any]) -> Optional[set[str]]:
        tool_access = dict(reg.get("tool_access") or {})
        hidden = {
            str(item or "").strip()
            for item in list(tool_access.get("hidden_allowed_tool_names") or [])
            if str(item or "").strip()
        }
        return hidden or None

    @staticmethod
    def _tools_view_from_payload(payload: Optional[Dict[str, Any]]) -> Optional[ToolsView]:
        row = dict(payload or {})
        if not row:
            return None
        return ToolsView(
            view_id=str(row.get("view_id") or "").strip() or "hosted-tools-view",
            mode=str(row.get("mode") or "").strip() or "advertised",
            allowed_tools=set(str(item or "").strip() for item in list(row.get("allowed_tools") or []) if str(item or "").strip()),
            advertised_tools=set(str(item or "").strip() for item in list(row.get("advertised_tools") or []) if str(item or "").strip()),
            hidden_allowed_tools=set(
                str(item or "").strip() for item in list(row.get("hidden_allowed_tools") or []) if str(item or "").strip()
            ),
            disabled_tools=set(str(item or "").strip() for item in list(row.get("disabled_tools") or []) if str(item or "").strip()),
            gated_tools=set(str(item or "").strip() for item in list(row.get("gated_tools") or []) if str(item or "").strip()),
            tool_constraints={
                str(tool_name or "").strip(): json.loads(json.dumps(dict(item or {})))
                for tool_name, item in dict(row.get("tool_constraints") or {}).items()
                if str(tool_name or "").strip() and isinstance(item, dict)
            },
        )

    @staticmethod
    def _registration_tool_routes(reg: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        tool_access = dict(reg.get("tool_access") or {})
        routes = dict(tool_access.get("tool_routes") or {})
        out: Dict[str, Dict[str, Any]] = {}
        for raw_name, raw_meta in routes.items():
            name = str(raw_name or "").strip()
            if not name:
                continue
            out[name] = dict(raw_meta or {})
        return out

    @staticmethod
    def _registration_toolbox_id(reg: Dict[str, Any]) -> str:
        bundle = dict(reg.get("bundle") or {})
        return str(bundle.get("toolbox_id") or bundle.get("bundle_id") or "").strip()

    @staticmethod
    def _callback_context_payload(context: Any) -> Dict[str, Any]:
        return {
            "engine_id": str(getattr(context, "engine_id", "") or "").strip() or None,
            "toolbox_id": str(getattr(context, "toolbox_id", "") or "").strip() or None,
            "tool_name": str(getattr(context, "tool_name", "") or "").strip() or None,
            "tool_call_id": str(getattr(context, "tool_call_id", "") or "").strip() or None,
            "tool_arguments": dict(getattr(context, "tool_arguments", {}) or {}),
        }

    def _toolbox_host_capability_dispatch_binding(
        self,
        *,
        engine_id: str,
        toolbox_id: str,
        tool_name: str,
        tool_call_id: str,
        tool_arguments: Dict[str, Any],
        sandbox_policy: Dict[str, Any],
        callback_binding: Optional[Dict[str, Any]] = None,
        host_api_approval: Optional[Dict[str, Any]] = None,
    ) -> Tuple[_HostedToolCallbackRelay, Dict[str, Any]]:
        original_binding = dict(callback_binding or {}) if isinstance(callback_binding, dict) else {}
        eid = str(engine_id or "").strip()
        relay = _HostedToolCallbackRelay()

        def _forward_callback(*, callback_name: str, payload: Any, context: Any) -> Dict[str, Any]:
            if not original_binding:
                return {"status": "error", "message": "callback_binding_missing"}
            from ..toolbox_executor_ipc import _invoke_callback_binding

            response = _invoke_callback_binding(
                original_binding,
                callback_name=str(callback_name or "").strip(),
                payload=payload,
                context=self._callback_context_payload(context),
            )
            return dict(response.get("result") or {}) if isinstance(response.get("result"), dict) else {"result": response.get("result")}

        def _approval_requester(payload: Dict[str, Any]) -> Dict[str, Any]:
            if not original_binding:
                return {"status": "denied", "approved": False, "decision": "deny", "reason": "approval_requester_unavailable"}
            from ..toolbox_executor_ipc import _invoke_callback_binding

            response = _invoke_callback_binding(
                original_binding,
                callback_name=HOST_CAPABILITY_APPROVAL_CALLBACK_NAME,
                payload=host_capability_approval_request(dict(payload or {})),
                context=dict(dict(payload or {}).get("context") or {}),
            )
            return dict(response.get("result") or response or {})

        def _dispatch_host_capability(payload: Dict[str, Any], context: Any) -> Dict[str, Any]:
            row = dict(payload or {})
            method = str(row.get("method") or "").strip()
            arguments = dict(row.get("arguments") or {}) if isinstance(row.get("arguments"), dict) else {}
            approval = dict(row.get("approval") or host_api_approval or {}) if isinstance(row.get("approval") or host_api_approval, dict) else {}
            callback_context = dict(arguments.get("callback_context") or {}) if isinstance(arguments.get("callback_context"), dict) else {}
            broker = HostCapabilityBroker(
                request_id=str(callback_context.get("tool_call_id") or getattr(context, "tool_call_id", "") or tool_call_id or ""),
                workflow_id=str(callback_context.get("workflow_id") or ""),
                package_id=str(callback_context.get("package_id") or ""),
                instance_id=str(callback_context.get("instance_id") or ""),
                engine_id=eid,
                consumer_id=eid,
                runtime_kind="toolbox_worker",
                policy=dict(sandbox_policy or {}),
                provider_invoker=self._host_capability_provider_invoker,
                approval_requester=_approval_requester if approval else None,
                audit_emitter=self._append_host_capability_audit_event,
            )
            broker.register_session(
                service_broker_host_capability_session(
                    session_id=f"{eid}.service_broker",
                    provider_id="builtin.service_broker",
                    owner="service",
                    visibility="consumer",
                    scope={"consumer_id": eid},
                    approval=approval,
                    binding={"engine_id": eid},
                )
            )
            result = broker.dispatch({"method": method, "arguments": arguments})
            return {"status": "ok", "result": dict(result or {})}

        def _processor(*, callback_name: str, payload: Any, context: Any) -> Dict[str, Any]:
            name = str(callback_name or "").strip()
            if name == HOST_CAPABILITY_DISPATCH_CALLBACK_NAME:
                return _dispatch_host_capability(dict(payload or {}) if isinstance(payload, dict) else {}, context)
            return _forward_callback(callback_name=name, payload=payload, context=context)

        binding = relay.bind_session(
            processor=_processor,
            toolbox_id=str(toolbox_id or "").strip(),
            tool_name=str(tool_name or "").strip(),
            tool_call_id=str(tool_call_id or "").strip(),
            tool_arguments=dict(tool_arguments or {}),
            callback_signature={"callbacks": [{"name": HOST_CAPABILITY_DISPATCH_CALLBACK_NAME, "payload_type": "object"}]},
            user_context=None,
        )
        return relay, binding

    def _toolbox_executor_registrations(self, toolbox_id: str) -> List[Dict[str, Any]]:
        tid = str(toolbox_id or "").strip()
        if not tid:
            return []
        rows: List[Dict[str, Any]] = []
        for row in self._read_engines():
            reg = dict(row or {})
            if str(reg.get("executor_kind") or "").strip() != "toolbox_executor":
                continue
            if self._registration_toolbox_id(reg) != tid:
                continue
            rows.append(reg)
        return rows

    def _active_toolbox_v2_snapshot(self, toolbox_id: str) -> Optional[Dict[str, Any]]:
        repository = getattr(self, "_toolbox_state_v2", None)
        if repository is None:
            return None
        return repository.get(str(toolbox_id or "").strip())

    def _active_toolbox_v2_registrations(self, toolbox_id: str) -> Optional[List[Dict[str, Any]]]:
        snapshot = self._active_toolbox_v2_snapshot(toolbox_id)
        if snapshot is None:
            return None
        engine_ids = sorted(
            {
                str(dict(route or {}).get("engine_id") or "").strip()
                for route in dict(snapshot.get("tool_routes") or {}).values()
            }
            - {""}
        )
        registrations: List[Dict[str, Any]] = []
        for engine_id in engine_ids:
            reg = dict(self._find_registration(engine_id) or {})
            if not reg or self._registration_toolbox_id(reg) != str(toolbox_id or "").strip():
                raise RuntimeError(f"toolbox_active_route_registration_missing:{toolbox_id}:{engine_id}")
            registrations.append(reg)
        return registrations

    def _cleanup_toolbox_bundle_root(self, reg: Dict[str, Any]) -> None:
        bundle = dict(reg.get("bundle") or {})
        raw = str(bundle.get("bundle_root") or "").strip()
        if not raw:
            return
        root = Path(raw).expanduser().resolve()
        allowed_root = (self.hosting_root / "toolbox_bundles").resolve()
        try:
            if root != allowed_root and allowed_root not in root.parents:
                return
        except Exception:
            return
        shutil.rmtree(root, ignore_errors=True)

    def _retire_toolbox_registration(self, engine_id: str) -> None:
        reg = self._find_registration(engine_id)
        if reg:
            try:
                self.shutdown(engine_id, timeout_seconds=2.0)
            except Exception:
                pass
            self.remove_registration(engine_id)
            self._cleanup_toolbox_bundle_root(reg)

    def _merge_toolbox_auto_requests(
        self,
        *,
        toolbox_id: str,
        incoming_requests: Optional[List[Dict[str, Any]]] = None,
        remove_keys: Optional[List[str]] = None,
    ) -> Tuple[List[Any], Dict[str, Any], Dict[str, Any]]:
        from ..toolbox_harness import ToolboxAutoAssignmentRequest

        tid = str(toolbox_id or "").strip()
        if not tid:
            raise ValueError("toolbox_id is required")
        state = self._read_toolboxes()
        toolboxes = dict(state.get("toolboxes") or {})
        toolbox_row = dict(toolboxes.get(tid) or {})
        persisted_requests = [
            ToolboxAutoAssignmentRequest.from_runtime_dict(dict(item or {}))
            for item in list(toolbox_row.get("requests") or [])
        ]
        merged: Dict[str, ToolboxAutoAssignmentRequest] = {
            req.stable_key(): req for req in persisted_requests
        }
        for row in list(incoming_requests or []):
            req = ToolboxAutoAssignmentRequest.from_runtime_dict(dict(row or {}))
            merged[req.stable_key()] = req
        for key in list(remove_keys or []):
            merged.pop(str(key or "").strip(), None)
        return list(merged.values()), state, toolboxes

    def _merge_toolbox_manual_requests(
        self,
        *,
        toolbox_id: str,
        incoming_requests: Optional[List[Dict[str, Any]]] = None,
        remove_keys: Optional[List[str]] = None,
    ) -> Tuple[List[Any], Dict[str, Any], Dict[str, Any]]:
        from ..toolbox_harness import ToolboxManualAssignmentRequest

        tid = str(toolbox_id or "").strip()
        if not tid:
            raise ValueError("toolbox_id is required")
        state = self._read_toolboxes()
        toolboxes = dict(state.get("toolboxes") or {})
        toolbox_row = dict(toolboxes.get(tid) or {})
        persisted_requests = [
            ToolboxManualAssignmentRequest.from_runtime_dict(dict(item or {}))
            for item in list(toolbox_row.get("manual_requests") or [])
        ]
        merged: Dict[str, ToolboxManualAssignmentRequest] = {
            req.stable_key(): req for req in persisted_requests
        }
        for row in list(incoming_requests or []):
            req = ToolboxManualAssignmentRequest.from_runtime_dict(dict(row or {}))
            merged[req.stable_key()] = req
        for key in list(remove_keys or []):
            merged.pop(str(key or "").strip(), None)
        return list(merged.values()), state, toolboxes

    @staticmethod
    def _normalize_intrinsic_tool_names(
        names: List[str],
        *,
        include_guides: bool = False,
    ) -> List[str]:
        out: List[str] = []
        seen: set[str] = set()
        base_names: List[str] = []
        for item in list(names or []):
            name = str(item or "").strip()
            if not name:
                continue
            if name not in seen:
                seen.add(name)
                out.append(name)
            if not name.endswith("_guide"):
                base_names.append(name)
        if include_guides:
            for base in base_names:
                guide_name = f"{base}_guide"
                if guide_name not in seen:
                    seen.add(guide_name)
                    out.append(guide_name)
        return out

    @staticmethod
    def _registration_sandbox_profile_id(reg: Dict[str, Any]) -> str:
        return str(dict(reg.get("bundle") or {}).get("sandbox_profile_id") or "default").strip() or "default"

    def _route_toolbox_registration(self, *, toolbox_id: str, tool_name: str, command_label: str) -> Dict[str, Any]:
        tid = str(toolbox_id or "").strip()
        name = str(tool_name or "").strip()
        if not tid:
            raise ValueError("toolbox_id is required")
        if not name:
            raise ValueError("tool_name is required")
        snapshot = self._active_toolbox_v2_snapshot(tid)
        if snapshot is not None:
            route = dict(dict(snapshot.get("tool_routes") or {}).get(name) or {})
            if not route:
                raise PermissionError(f"tool_not_allowed:{name}")
            engine_id = str(route.get("engine_id") or "").strip()
            profile_id = str(route.get("profile_id") or "").strip()
            reg = dict(self._find_registration(engine_id) or {})
            bundle = dict(reg.get("bundle") or {})
            if (
                not reg
                or self._registration_toolbox_id(reg) != tid
                or str(bundle.get("resolved_profile_id") or bundle.get("sandbox_profile_id") or "") != profile_id
            ):
                raise RuntimeError(f"toolbox_active_route_registration_mismatch:{tid}:{name}")
            return self._require_toolbox_executor_registration(engine_id, command_label=command_label)
        matches: List[Dict[str, Any]] = []
        for reg in self._toolbox_executor_registrations(tid):
            if str(reg.get("routing_state") or "active") != "active":
                continue
            allowed = self._registration_allowed_tool_names(reg)
            if allowed is not None and name in allowed:
                matches.append(reg)
        if not matches:
            raise PermissionError(f"tool_not_allowed:{name}")
        if len(matches) > 1:
            raise RuntimeError(f"toolbox_route_ambiguous:{tid}:{name}")
        return self._require_toolbox_executor_registration(
            str(matches[0].get("engine_id") or ""),
            command_label=command_label,
        )

    def _require_toolbox_executor_registration(self, engine_id: str, *, command_label: str) -> Dict[str, Any]:
        reg = self._require_ipc_registration(engine_id, command_label=command_label)
        executor_kind = str(reg.get("executor_kind") or "").strip()
        if executor_kind and executor_kind != "toolbox_executor":
            raise ValueError(f"{command_label} is only supported for toolbox executors")
        return reg

    def _toolbox_runtime_base(self) -> HostedToolboxRuntimeBase:
        from ..sandbox.toolbox_runtime import HostedToolboxRuntimeBase

        base = getattr(self, "_hosted_toolbox_runtime_base", None)
        if not isinstance(base, HostedToolboxRuntimeBase):
            base = HostedToolboxRuntimeBase()
            setattr(self, "_hosted_toolbox_runtime_base", base)
        return base

    @staticmethod
    def _toolbox_registration_environment_key(reg: Dict[str, Any]) -> str:
        env = dict(dict(reg or {}).get("environment") or {})
        caps = dict(dict(reg or {}).get("capabilities") or {})
        return str(env.get("environment_key") or caps.get("environment_key") or dict(reg or {}).get("engine_id") or "").strip()

    @staticmethod
    def _toolbox_registration_capacity(reg: Dict[str, Any]) -> int:
        caps = dict(dict(reg or {}).get("capabilities") or {})
        for key in ("capacity", "max_concurrency", "max_parallel_calls"):
            try:
                value = int(caps.get(key) or 0)
            except Exception:
                value = 0
            if value > 0:
                return max(1, min(value, 1024))
        return 256

    @classmethod
    def _toolbox_registration_described_capacity(cls, reg: Dict[str, Any]) -> int:
        """Return configured capacity for discovery before a runtime pool exists."""
        caps = dict(dict(reg or {}).get("capabilities") or {})
        for key in ("capacity", "max_concurrency", "max_parallel_calls"):
            if key not in caps or caps.get(key) is None:
                continue
            try:
                value = int(caps.get(key))
            except Exception:
                return cls._toolbox_registration_capacity(reg)
            return max(0, min(value, 1024))
        return cls._toolbox_registration_capacity(reg)

    @staticmethod
    def _toolbox_registration_queue_config(reg: Dict[str, Any]) -> Dict[str, Any]:
        caps = dict(dict(reg or {}).get("capabilities") or {})
        policy = str(caps.get("queue_policy") or caps.get("concurrency_queue_policy") or "bounded").strip().lower()
        if policy not in {"bounded", "fail_fast"}:
            policy = "bounded"

        depth = 32
        for key in ("queue_depth", "max_queue_depth"):
            if key not in caps or caps.get(key) is None:
                continue
            try:
                depth = int(caps.get(key))
            except Exception:
                depth = 32
            break

        timeout = 30.0
        for key in ("queue_timeout_seconds", "queue_wait_timeout_seconds"):
            if key not in caps or caps.get(key) is None:
                continue
            try:
                timeout = float(caps.get(key))
            except Exception:
                timeout = 30.0
            break
        return {
            "queue_policy": policy,
            "queue_depth": max(0, min(depth, 4096)),
            "queue_timeout_seconds": max(0.0, min(timeout, 3600.0)),
        }

    def _toolbox_tool_concurrency_policy(
        self,
        *,
        toolbox_id: str,
        tool_name: str,
        call: Dict[str, Any],
    ) -> Dict[str, Any]:
        tid = str(toolbox_id or "").strip()
        name = str(tool_name or "").strip()
        state = self._read_toolboxes() if tid else {}
        row = dict(dict(state.get("toolboxes") or {}).get(tid) or {})
        metadata = dict(self._toolbox_tool_metadata(row).get(name) or {})
        raw = dict(metadata.get("concurrency") or {})
        mode = str(raw.get("mode") or "parallel").strip().lower()
        if mode not in {"parallel", "serial", "keyed", "exclusive"}:
            mode = "parallel"
        group = str(raw.get("group") or "").strip()
        if mode in {"serial", "keyed"} and not group:
            group = name or "tool"
        if mode == "exclusive" and not group:
            group = "toolbox"
        arguments = dict(call.get("arguments") or {}) if isinstance(call.get("arguments"), dict) else {}
        resource_key = str(raw.get("resource_key") or "").strip()
        key_arguments = raw.get("key_arguments") or raw.get("resource_key_arguments")
        if not resource_key:
            key_argument = str(raw.get("key_argument") or raw.get("resource_key_argument") or "").strip()
            if key_argument:
                key_arguments = [key_argument]
            if isinstance(key_arguments, str):
                key_arguments = [key_arguments]
            if mode == "keyed":
                values: List[Any] = []
                for key in list(key_arguments or []):
                    current: Any = arguments
                    for part in str(key or "").split("."):
                        if isinstance(current, dict):
                            current = current.get(part)
                        else:
                            current = None
                    values.append(current)
                if not values:
                    values = [arguments.get("resource_key", arguments.get("key", "__missing__"))]
                resource_key = json.dumps(values, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        try:
            max_concurrency = int(raw.get("max_concurrency") or 0)
        except Exception:
            max_concurrency = 0
        if mode == "serial":
            max_concurrency = 1
        if max_concurrency > 0 and not group:
            group = name or "tool"
        return {
            "mode": mode,
            "group": group,
            "resource_key": resource_key,
            "max_concurrency": max(0, max_concurrency),
            "decision": "compatibility_default" if not raw else "declared",
        }

    def _toolbox_worker_slot(self, *, reg: Dict[str, Any], environment_key: str, capacity: int) -> Any:
        from ..sandbox.toolbox_runtime import HostedToolboxRuntimeBase

        return HostedToolboxRuntimeBase.worker_slot(
            engine_id=str(dict(reg or {}).get("engine_id") or "").strip(),
            environment_key=str(environment_key or "").strip(),
            capacity=capacity,
            pid=int(dict(reg or {}).get("pid") or 0) or None,
            status="registered",
        )

    def _toolbox_pool_resources(self, reg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        environment_key = self._toolbox_registration_environment_key(reg)
        if not environment_key:
            return None
        resources = self._toolbox_runtime_base().resources(environment_key)
        return dict(resources or {}) if str(dict(resources or {}).get("status") or "") != "not_found" else None

    def toolbox_describe(
        self,
        *,
        engine_id: str = "",
        toolbox_id: str = "",
        timeout_seconds: float = 10.0,
    ) -> Dict[str, Any]:
        eid = str(engine_id or "").strip()
        tid = str(toolbox_id or "").strip()
        if not eid and not tid:
            raise ValueError("engine_id or toolbox_id is required")
        if tid and not eid:
            v2_regs = self._active_toolbox_v2_registrations(tid)
            regs = v2_regs if v2_regs is not None else self._toolbox_executor_registrations(tid)
            snapshot = self._active_toolbox_v2_snapshot(tid) if v2_regs is not None else None
            if not regs and snapshot is None:
                raise ValueError(f"toolbox '{tid}' has no registered sandbox executors")
            state = self._read_toolboxes()
            toolbox_row = dict(dict(state.get("toolboxes") or {}).get(tid) or {})
            tool_names: set[str] = set()
            advertised_tool_names: set[str] = set()
            hidden_allowed_tool_names: set[str] = set()
            sandbox_profile_ids: set[str] = set()
            engine_ids: List[str] = []
            hosted_pools: Dict[str, Any] = {}
            parallel_rows: List[Dict[str, Any]] = []
            for reg in regs:
                reg_engine_id = str(reg.get("engine_id") or "")
                engine_ids.append(reg_engine_id)
                pool = self._toolbox_pool_resources(reg)
                if pool is not None and reg_engine_id:
                    hosted_pools[reg_engine_id] = pool
                registration_queue = self._toolbox_registration_queue_config(reg)
                registration_capacity = self._toolbox_registration_described_capacity(reg)
                metrics = dict(pool.get("metrics") or {}) if pool is not None else {}
                parallel_rows.append(
                    {
                        "effective_max_concurrency": int(metrics["desired_capacity"])
                        if pool is not None and "desired_capacity" in metrics
                        else registration_capacity,
                        "queue_policy": str(metrics["queue_policy"])
                        if pool is not None and "queue_policy" in metrics
                        else registration_queue["queue_policy"],
                        "queue_depth": int(metrics["queue_depth"])
                        if pool is not None and "queue_depth" in metrics
                        else registration_queue["queue_depth"],
                        "queue_timeout_seconds": float(metrics["queue_timeout_seconds"])
                        if pool is not None and "queue_timeout_seconds" in metrics
                        else registration_queue["queue_timeout_seconds"],
                        "active_calls": int(metrics.get("active_calls") or 0),
                        "queued_calls": int(metrics.get("queued_calls") or 0),
                        "worker_process_count": int(metrics.get("worker_count") or 0),
                    }
                )
                for name in list(self._registration_allowed_tool_names(reg) or set()):
                    tool_names.add(name)
                for name in list(self._registration_advertised_tool_names(reg) or set()):
                    advertised_tool_names.add(name)
                for name in list(self._registration_hidden_allowed_tool_names(reg) or set()):
                    hidden_allowed_tool_names.add(name)
                sandbox_profile_ids.add(str(dict(reg.get("bundle") or {}).get("sandbox_profile_id") or "default"))
            if snapshot is not None:
                tool_names = set(dict(snapshot.get("tool_routes") or {}))
                advertised_tool_names.intersection_update(tool_names)
                hidden_allowed_tool_names.intersection_update(tool_names)
            return {
                "status": "ok",
                "toolbox_id": tid,
                "engine_ids": [eid for eid in engine_ids if eid],
                "all_registered_tool_names": sorted(tool_names),
                "allowed_tool_names": sorted(tool_names),
                "advertised_tool_names": sorted(advertised_tool_names or tool_names),
                "hidden_allowed_tool_names": sorted(hidden_allowed_tool_names),
                "tool_metadata": self._toolbox_tool_metadata(toolbox_row),
                "sandbox_profile_ids": sorted([pid for pid in sandbox_profile_ids if pid]),
                "executor_kind": "toolbox_executor",
                "mode": "sandbox",
                "parallel_execution": {
                    "async_within_executor": True,
                    "sandbox_pool": len(engine_ids) > 1,
                    "supported": True,
                    "effective_max_concurrency": max(
                        [int(row["effective_max_concurrency"]) for row in parallel_rows] or [0]
                    ),
                    "queue_policy": "bounded"
                    if any(row["queue_policy"] == "bounded" for row in parallel_rows)
                    else "fail_fast",
                    "queue_depth": max(
                        [int(row["queue_depth"]) for row in parallel_rows] or [0]
                    ),
                    "queue_timeout_seconds": max(
                        [float(row["queue_timeout_seconds"]) for row in parallel_rows] or [0.0]
                    ),
                    "active_calls": sum(
                        [int(row["active_calls"]) for row in parallel_rows]
                    ),
                    "queued_calls": sum(
                        [int(row["queued_calls"]) for row in parallel_rows]
                    ),
                    "worker_process_count": sum(
                        [int(row["worker_process_count"]) for row in parallel_rows]
                    ),
                    "execution_model": "threaded_worker",
                },
                "hosted_pools": hosted_pools,
            }
        reg = self._require_toolbox_executor_registration(eid, command_label="toolbox-describe")
        out = self._ipc_call(
            reg=reg,
            payload={"kind": "rpc_call", "engine_id": eid, "method": "toolbox.describe", "params": {}},
            timeout_seconds=float(timeout_seconds or 10.0),
        )
        if str(out.get("status") or "").strip().lower() == "error":
            raise RuntimeError(str(out.get("message") or "toolbox_describe_failed"))
        result = dict(out or {})
        result.setdefault("engine_id", eid)
        result.setdefault("executor_kind", str(reg.get("executor_kind") or "toolbox_executor"))
        result.setdefault("bundle", dict(reg.get("bundle") or {}))
        result.setdefault("tool_access", dict(reg.get("tool_access") or {}))
        result.setdefault("all_registered_tool_names", sorted(list(self._registration_allowed_tool_names(reg) or set())))
        result.setdefault("allowed_tool_names", sorted(list(self._registration_allowed_tool_names(reg) or set())))
        result.setdefault("advertised_tool_names", sorted(list(self._registration_advertised_tool_names(reg) or set())))
        result.setdefault("hidden_allowed_tool_names", sorted(list(self._registration_hidden_allowed_tool_names(reg) or set())))
        pool = self._toolbox_pool_resources(reg)
        if pool is not None:
            result.setdefault("hosted_pool", pool)
            result.setdefault("toolbox_pool", pool)
        metrics = dict(pool.get("metrics") or {}) if pool is not None else {}
        registration_capacity = self._toolbox_registration_described_capacity(reg)
        registration_queue = self._toolbox_registration_queue_config(reg)
        parallel = dict(result.get("parallel_execution") or {})
        effective_max = (
            int(metrics["desired_capacity"])
            if pool is not None and "desired_capacity" in metrics
            else int(parallel.get("effective_max_concurrency") or 0) or registration_capacity
        )
        queue_depth = (
            int(metrics["queue_depth"])
            if pool is not None and "queue_depth" in metrics
            else int(parallel.get("queue_depth") or 0) or registration_queue["queue_depth"]
        )
        queue_timeout = (
            float(metrics["queue_timeout_seconds"])
            if pool is not None and "queue_timeout_seconds" in metrics
            else float(parallel.get("queue_timeout_seconds") or 0.0) or registration_queue["queue_timeout_seconds"]
        )
        parallel.update(
            {
                "supported": bool(parallel.get("supported", True)),
                "async_within_executor": bool(parallel.get("async_within_executor", True)),
                "sandbox_pool": bool(parallel.get("sandbox_pool", False)),
                "effective_max_concurrency": effective_max,
                "queue_policy": str(parallel.get("queue_policy") or metrics.get("queue_policy") or registration_queue["queue_policy"]),
                "queue_depth": queue_depth,
                "queue_timeout_seconds": queue_timeout,
                "active_calls": int(parallel.get("active_calls") or metrics.get("active_calls") or 0),
                "queued_calls": int(parallel.get("queued_calls") or metrics.get("queued_calls") or 0),
                "worker_process_count": int(parallel.get("worker_process_count") or metrics.get("worker_count") or 0),
                "execution_model": str(parallel.get("execution_model") or "threaded_worker"),
            }
        )
        result["parallel_execution"] = parallel
        result.pop("tool_names", None)
        return result

    def toolbox_gate(
        self,
        *,
        engine_id: str = "",
        toolbox_id: str = "",
        tool_name: str,
        tools_view: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        eid = str(engine_id or "").strip()
        tid = str(toolbox_id or "").strip()
        name = str(tool_name or "").strip()
        view = self._tools_view_from_payload(tools_view)
        if not name:
            raise ValueError("tool_name is required")
        if not eid and not tid:
            raise ValueError("engine_id or toolbox_id is required")
        if tid and not eid:
            v2_snapshot = self._active_toolbox_v2_snapshot(tid)
            regs = (
                self._active_toolbox_v2_registrations(tid)
                if v2_snapshot is not None
                else self._toolbox_executor_registrations(tid)
            )
            if not regs and v2_snapshot is None:
                return {
                    "status": "ok",
                    "toolbox_id": tid,
                    "tool_name": name,
                    "outcome": "unavailable_backend",
                    "reason": "toolbox_executor_missing",
                    "executable": False,
                    "requires_confirmation": False,
                    "backend": "sandbox",
                }
            allowed_for_toolbox: set[str] = (
                set(dict(v2_snapshot.get("tool_routes") or {}))
                if v2_snapshot is not None
                else set()
            )
            if v2_snapshot is None:
                for item in regs:
                    allowed_for_toolbox.update(self._registration_allowed_tool_names(item) or set())
            if view is not None and name in allowed_for_toolbox and view.is_gated(name):
                return {
                    "status": "ok",
                    "toolbox_id": tid,
                    "tool_name": name,
                    "outcome": "gated_requires_confirmation",
                    "reason": "gated_requires_confirmation",
                    "executable": False,
                    "requires_confirmation": True,
                    "backend": "sandbox",
                }
            if view is not None and name in allowed_for_toolbox and not view.is_allowed(name):
                return {
                    "status": "ok",
                    "toolbox_id": tid,
                    "tool_name": name,
                    "outcome": "denied",
                    "reason": "blocked_in_scope",
                    "executable": False,
                    "requires_confirmation": False,
                    "backend": "sandbox",
                }
            try:
                reg = self._route_toolbox_registration(toolbox_id=tid, tool_name=name, command_label="toolbox-gate")
            except PermissionError:
                return {
                    "status": "ok",
                    "toolbox_id": tid,
                    "tool_name": name,
                    "outcome": "denied",
                    "reason": "tool_not_allowed",
                    "executable": False,
                    "requires_confirmation": False,
                    "backend": "sandbox",
                }
            eid = str(reg.get("engine_id") or "").strip()
        else:
            reg = self._require_toolbox_executor_registration(eid, command_label="toolbox-gate")
            allowed_tool_names = self._registration_allowed_tool_names(reg)
            if allowed_tool_names is not None and name not in allowed_tool_names:
                return {
                    "status": "ok",
                    "engine_id": eid,
                    "tool_name": name,
                    "outcome": "denied",
                    "reason": "tool_not_allowed",
                    "executable": False,
                    "requires_confirmation": False,
                    "backend": "sandbox",
                }
            if view is not None and allowed_tool_names is not None and name in allowed_tool_names and view.is_gated(name):
                return {
                    "status": "ok",
                    "engine_id": eid,
                    "toolbox_id": tid or self._registration_toolbox_id(reg),
                    "tool_name": name,
                    "outcome": "gated_requires_confirmation",
                    "reason": "gated_requires_confirmation",
                    "executable": False,
                    "requires_confirmation": True,
                    "backend": "sandbox",
                }
            if view is not None and allowed_tool_names is not None and name in allowed_tool_names and not view.is_allowed(name):
                return {
                    "status": "ok",
                    "engine_id": eid,
                    "toolbox_id": tid or self._registration_toolbox_id(reg),
                    "tool_name": name,
                    "outcome": "denied",
                    "reason": "blocked_in_scope",
                    "executable": False,
                    "requires_confirmation": False,
                    "backend": "sandbox",
                }
        result = {
            "status": "ok",
            "engine_id": eid,
            "toolbox_id": tid or self._registration_toolbox_id(reg),
            "tool_name": name,
            "outcome": "allowed",
            "reason": "allowed",
            "executable": True,
            "requires_confirmation": False,
            "backend": "sandbox",
        }
        return result

    def toolbox_execute(
        self,
        *,
        engine_id: str = "",
        toolbox_id: str = "",
        tool_call: Dict[str, Any],
        timeout_seconds: float = 30.0,
        tools_view: Optional[Dict[str, Any]] = None,
        callback_binding: Optional[Dict[str, Any]] = None,
        host_api_approval: Optional[Dict[str, Any]] = None,
        execution_request_id: str = "",
        owner_actor_id: str = "service:local",
    ) -> Dict[str, Any]:
        eid = str(engine_id or "").strip()
        call = dict(tool_call or {})
        tool_name = str(call.get("name") or "").strip()
        view = self._tools_view_from_payload(tools_view)
        if not tool_name:
            raise ValueError("tool_call.name is required")
        tid = str(toolbox_id or "").strip()
        if not eid and not tid:
            raise ValueError("engine_id or toolbox_id is required")
        if tid and not eid:
            gate = self.toolbox_gate(toolbox_id=tid, tool_name=tool_name, tools_view=tools_view)
            if str(gate.get("outcome") or "").strip().lower() != "allowed":
                reason = str(gate.get("reason") or gate.get("outcome") or "denied")
                raise PermissionError(f"{reason}:{tool_name}")
            reg = self._route_toolbox_registration(toolbox_id=tid, tool_name=tool_name, command_label="toolbox-execute")
            eid = str(reg.get("engine_id") or "").strip()
        else:
            reg = self._require_toolbox_executor_registration(eid, command_label="toolbox-execute")
            allowed_tool_names = self._registration_allowed_tool_names(reg)
            if allowed_tool_names is not None and tool_name not in allowed_tool_names:
                raise PermissionError(f"tool_not_allowed:{tool_name}")
            if view is not None and allowed_tool_names is not None and tool_name in allowed_tool_names and view.is_gated(tool_name):
                raise PermissionError(f"gated_requires_confirmation:{tool_name}")
            if view is not None and allowed_tool_names is not None and tool_name in allowed_tool_names and not view.is_allowed(tool_name):
                raise PermissionError(f"blocked_in_scope:{tool_name}")
        environment_key = self._toolbox_registration_environment_key(reg)
        capacity = self._toolbox_registration_capacity(reg)
        toolbox_identity = tid or self._registration_toolbox_id(reg)
        concurrency = self._toolbox_tool_concurrency_policy(
            toolbox_id=toolbox_identity,
            tool_name=tool_name,
            call=call,
        )
        queue_config = self._toolbox_registration_queue_config(reg)
        model_tool_call_id = str(call.get("id") or call.get("tool_call_id") or "").strip() or f"tool-call-{uuid.uuid4().hex}"
        request_id = str(execution_request_id or "").strip()
        if not request_id:
            raise ValueError("execution_request_id is required for durable hosted execution")
        receipt_namespace = self._toolbox_operation_namespace(
            engine_id=eid if not tid else "",
            toolbox_id=tid,
        )
        selector = HostedOperationSelector(
            kind="toolbox_id" if tid else "engine_id",
            id=toolbox_identity if tid else eid,
        )
        fingerprint = hosted_execution_fingerprint(
            {
                "execution_kind": HostedExecutionKind.TOOLBOX.value,
                "selector": selector.to_dict(),
                "tool": {
                    "name": tool_name,
                    "arguments": call.get("arguments") if isinstance(call.get("arguments"), dict) else {},
                },
                "policy": {
                    "tools_view": dict(tools_view or {}) if isinstance(tools_view, dict) else None,
                    "host_api_approval": dict(host_api_approval or {}) if isinstance(host_api_approval, dict) else None,
                    "sandbox_policy": dict(reg.get("sandbox_policy") or {}),
                    "sandbox_profile_id": self._registration_sandbox_profile_id(reg),
                    "concurrency": dict(concurrency),
                },
            }
        )
        prepared = self._hosted_operations.prepare(
            owner_actor_id=str(owner_actor_id or "service:local").strip() or "service:local",
            execution_kind=HostedExecutionKind.TOOLBOX,
            selector=selector,
            namespace=receipt_namespace,
            request_id=request_id,
            fingerprint=fingerprint,
            metadata={
                "engine_id": eid,
                "toolbox_id": toolbox_identity,
                "tool_name": tool_name,
                "tool_call_id": model_tool_call_id,
                "environment_key": environment_key,
                "retain_terminal_result": bool(dict(reg.get("sandbox_policy") or {}).get("retain_terminal_result")),
            },
        )
        operation_action = str(prepared.get("action") or "")
        prepared_status = dict(prepared.get("status") or {})
        if operation_action in {"conflict", "forgotten", "replay"}:
            return prepared_status
        if operation_action == "capacity":
            raise RuntimeError("hosted_operation_capacity_exceeded")
        operation_id = str(dict(prepared_status.get("operation") or {}).get("operation_id") or "")
        if operation_action == "attach":
            return self._hosted_operations.wait_for_terminal(
                operation_id=operation_id,
                timeout_seconds=float(timeout_seconds or 30.0),
            )
        base = self._toolbox_runtime_base()

        def _persist_terminal(envelope: Dict[str, Any], lifecycle: str) -> Dict[str, Any]:
            return self._hosted_operations.finish(
                operation_id=operation_id,
                lifecycle=lifecycle,
                envelope=envelope,
                reason=str(envelope.get("reason") or "").strip(),
            )

        scheduled = base.submit_request(
            environment_key=environment_key,
            request_id=request_id,
            profile=self._registration_sandbox_profile_id(reg),
            factory=lambda _key, cap: self._toolbox_worker_slot(reg=reg, environment_key=environment_key, capacity=cap),
            desired_capacity=capacity,
            operation_id=tool_name,
            input_bytes=len(json.dumps(call, ensure_ascii=False).encode("utf-8", errors="replace")),
            queue_policy=str(queue_config.get("queue_policy") or "bounded"),
            queue_depth=int(queue_config.get("queue_depth") or 0),
            queue_timeout_seconds=float(queue_config.get("queue_timeout_seconds") or 0.0),
            concurrency=concurrency,
        )
        if str(scheduled.get("status") or "") != "ok":
            pool_snapshot = base.resources(environment_key)
            request_snapshot = dict(scheduled.get("request") or {})
            failure_reason = str(scheduled.get("reason") or "capacity_exceeded")
            return _persist_terminal({
                "status": "error",
                "outcome": "error",
                "reason": failure_reason,
                "error": failure_reason,
                "engine_id": eid,
                "toolbox_id": tid or self._registration_toolbox_id(reg),
                "tool_name": tool_name,
                "tool_call_id": model_tool_call_id,
                "request_id": request_id,
                "environment_key": environment_key,
                "worker_id": request_snapshot.get("worker_id"),
                "retry_count": 0,
                "admission": request_snapshot.get("admission") or "rejected",
                "concurrency": dict(concurrency),
                "request": request_snapshot,
                "diagnostics": {
                    "request": request_snapshot,
                    "concurrency": dict(concurrency),
                    "pool": pool_snapshot,
                },
                "hosted_pool": pool_snapshot,
            }, "terminal_failure")
        dispatch_claim = base.claim_dispatch(environment_key=environment_key, request_id=request_id)
        if str(dispatch_claim.get("status") or "") != "ok":
            request_snapshot = dict(dispatch_claim.get("request") or {})
            pool_snapshot = base.resources(environment_key)
            return _persist_terminal({
                "status": "error",
                "outcome": "canceled",
                "reason": str(request_snapshot.get("reason") or "canceled"),
                "error": str(request_snapshot.get("reason") or "canceled"),
                "engine_id": eid,
                "toolbox_id": tid or self._registration_toolbox_id(reg),
                "tool_name": tool_name,
                "tool_call_id": model_tool_call_id,
                "request_id": request_id,
                "environment_key": environment_key,
                "worker_id": request_snapshot.get("worker_id"),
                "retry_count": 0,
                "admission": request_snapshot.get("admission") or "canceled",
                "concurrency": dict(concurrency),
                "request": request_snapshot,
                "diagnostics": {
                    "request": request_snapshot,
                    "concurrency": dict(concurrency),
                    "pool": pool_snapshot,
                },
                "hosted_pool": pool_snapshot,
            }, "terminal_cancellation")
        finished = False
        dispatch_relay: Optional[_HostedToolCallbackRelay] = None
        dispatch_binding: Optional[Dict[str, Any]] = None
        try:
            dispatch_relay, dispatch_binding = self._toolbox_host_capability_dispatch_binding(
                engine_id=eid,
                toolbox_id=tid or self._registration_toolbox_id(reg),
                tool_name=tool_name,
                tool_call_id=model_tool_call_id,
                tool_arguments=call.get("arguments") if isinstance(call.get("arguments"), dict) else {},
                sandbox_policy=dict(reg.get("sandbox_policy") or {}),
                callback_binding=dict(callback_binding or {}) if isinstance(callback_binding, dict) else None,
                host_api_approval=dict(host_api_approval or {}) if isinstance(host_api_approval, dict) else None,
            )
            durable_dispatch = self._hosted_operations.mark_dispatch_claimed(operation_id=operation_id)
            if str(durable_dispatch.get("lifecycle") or "") != HostedOperationLifecycle.RUNNING.value:
                base.cancel_request(environment_key=environment_key, request_id=request_id)
                return durable_dispatch
            out = self._ipc_call(
                reg=reg,
                payload={
                    "kind": "rpc_call",
                    "engine_id": eid,
                    "method": "toolbox.execute",
                    "params": {
                        "tool_call": call,
                        "callback_binding": dict(dispatch_binding or {}) if isinstance(dispatch_binding, dict) else None,
                        "host_api_approval": dict(host_api_approval or {}) if isinstance(host_api_approval, dict) else None,
                    },
                },
                timeout_seconds=float(timeout_seconds or 30.0),
            )
            if str(out.get("status") or "").strip().lower() == "error":
                reason = str(out.get("message") or "toolbox_execute_failed")
                finish_out = base.finish_request(environment_key=environment_key, request_id=request_id, status="error", reason=reason)
                finished = True
                request_snapshot = dict(finish_out.get("request") or {})
                pool_snapshot = base.resources(environment_key)
                return _persist_terminal({
                    "status": "error",
                    "outcome": "error",
                    "reason": reason,
                    "error": str(out.get("message") or reason),
                    "engine_id": eid,
                    "toolbox_id": tid or self._registration_toolbox_id(reg),
                    "tool_name": tool_name,
                    "tool_call_id": model_tool_call_id,
                    "request_id": request_id,
                    "environment_key": environment_key,
                    "worker_id": request_snapshot.get("worker_id"),
                    "retry_count": 0,
                    "admission": request_snapshot.get("admission") or "admitted",
                    "concurrency": dict(concurrency),
                    "request": request_snapshot,
                    "diagnostics": {
                        "request": request_snapshot,
                        "concurrency": dict(concurrency),
                        "pool": pool_snapshot,
                    },
                    "hosted_pool": pool_snapshot,
                }, "terminal_failure")
            result = dict(out or {})
            output_bytes = len(json.dumps(result, ensure_ascii=False, default=str).encode("utf-8", errors="replace"))
            finish_out = base.finish_request(environment_key=environment_key, request_id=request_id, status="ok", output_bytes=output_bytes)
            finished = True
            request_snapshot = dict(finish_out.get("request") or {})
            pool_snapshot = base.resources(environment_key)
            result.setdefault("engine_id", eid)
            result.setdefault("toolbox_id", tid or self._registration_toolbox_id(reg))
            result.setdefault("tool_name", tool_name)
            result.setdefault("tool_call_id", model_tool_call_id)
            result.setdefault("request_id", request_id)
            result.setdefault("environment_key", environment_key)
            result.setdefault("worker_id", request_snapshot.get("worker_id"))
            result.setdefault("retry_count", 0)
            result.setdefault("admission", request_snapshot.get("admission") or "admitted")
            result.setdefault("concurrency", dict(concurrency))
            result.setdefault("request", request_snapshot)
            result.setdefault(
                "diagnostics",
                {
                    "request": request_snapshot,
                    "concurrency": dict(concurrency),
                    "pool": pool_snapshot,
                },
            )
            result.setdefault("hosted_pool", pool_snapshot)
            result.setdefault("toolbox_pool", result["hosted_pool"])
            return _persist_terminal(result, "terminal_success")
        except Exception as exc:
            reason = "toolbox_execute_timeout" if isinstance(exc, TimeoutError) else str(exc) or "toolbox_execute_failed"
            finish_status = "timeout" if isinstance(exc, TimeoutError) else "error"
            if not finished:
                finish_out = base.finish_request(
                    environment_key=environment_key,
                    request_id=request_id,
                    status=finish_status,
                    reason=reason,
                )
            else:
                finish_out = base.request_status(environment_key=environment_key, request_id=request_id)
            request_snapshot = dict(finish_out.get("request") or {})
            pool_snapshot = base.resources(environment_key)
            return _persist_terminal({
                "status": "timeout" if isinstance(exc, TimeoutError) else "error",
                "outcome": "timeout" if isinstance(exc, TimeoutError) else "error",
                "reason": reason,
                "error": str(exc) or reason,
                "error_type": type(exc).__name__,
                "engine_id": eid,
                "toolbox_id": tid or self._registration_toolbox_id(reg),
                "tool_name": tool_name,
                "tool_call_id": model_tool_call_id,
                "request_id": request_id,
                "environment_key": environment_key,
                "worker_id": request_snapshot.get("worker_id"),
                "retry_count": 0,
                "admission": request_snapshot.get("admission") or "admitted",
                "concurrency": dict(concurrency),
                "request": request_snapshot,
                "diagnostics": {
                    "request": request_snapshot,
                    "concurrency": dict(concurrency),
                    "pool": pool_snapshot,
                },
                "hosted_pool": pool_snapshot,
            }, "terminal_failure")
        finally:
            if dispatch_relay is not None and dispatch_binding:
                dispatch_relay.release_session(str(dispatch_binding.get("session_token") or ""))

    def _cancel_toolbox_operation(
        self,
        *,
        record: Dict[str, Any],
        reason: str = "client_requested",
        timeout_seconds: float = 8.0,
        respawn: bool = True,
    ) -> Dict[str, Any]:
        row = dict(record or {})
        operation = dict(row.get("operation") or {})
        metadata = dict(row.get("metadata") or {})
        operation_id = str(operation.get("operation_id") or "").strip()
        owner_actor_id = str(row.get("owner_actor_id") or "").strip()
        eid = str(metadata.get("engine_id") or "").strip()
        tid = str(metadata.get("toolbox_id") or "").strip()
        name = str(metadata.get("tool_name") or "").strip()
        model_tool_call_id = str(metadata.get("tool_call_id") or "").strip()
        call_id = str(operation.get("request_id") or "").strip()
        if not eid and not tid:
            raise ValueError("stored toolbox operation selector is invalid")
        lifecycle = HostedOperationLifecycle(str(row.get("lifecycle") or ""))
        if lifecycle in {
            HostedOperationLifecycle.QUEUED,
            HostedOperationLifecycle.INTERRUPTED_BEFORE_DISPATCH,
        }:
            canceled = self._hosted_operations.cancel_before_dispatch(
                operation_id=operation_id,
                reason=str(reason or "canceled_before_dispatch"),
            )
            if canceled is not None:
                return canceled
        if lifecycle in {
            HostedOperationLifecycle.TERMINAL_SUCCESS,
            HostedOperationLifecycle.TERMINAL_FAILURE,
            HostedOperationLifecycle.TERMINAL_CANCELLATION,
            HostedOperationLifecycle.INTERRUPTED_AFTER_DISPATCH_UNKNOWN,
            HostedOperationLifecycle.FORGOTTEN,
        }:
            return self._hosted_operations.status(ref=operation, owner_actor_id=owner_actor_id)

        def _cancel_failure(failure_reason: str) -> Dict[str, Any]:
            return self._hosted_operations.finish(
                operation_id=operation_id,
                lifecycle=HostedOperationLifecycle.TERMINAL_FAILURE,
                envelope={
                    "status": "error",
                    "outcome": "cancel_failed",
                    "reason": failure_reason,
                    "engine_id": eid or None,
                    "toolbox_id": tid or None,
                    "tool_name": name or None,
                    "request_id": call_id,
                },
                reason=failure_reason,
            )

        target_regs: List[Dict[str, Any]] = []
        if eid:
            reg = dict(self._find_registration(eid) or {})
            if not reg:
                return _cancel_failure("engine_not_found")
            target_regs = [reg]
        elif tid:
            if name:
                try:
                    target_regs = [self._route_toolbox_registration(toolbox_id=tid, tool_name=name, command_label="hosted-operation-cancel")]
                except PermissionError:
                    return _cancel_failure("tool_not_allowed")
            else:
                target_regs = list(self._toolbox_executor_registrations(tid))
            if not target_regs:
                return _cancel_failure("toolbox_executor_missing")

        canceled_engine_ids: List[str] = []
        failed_engine_ids: List[str] = []
        shutdown_results: Dict[str, Dict[str, Any]] = {}
        hosted_pool_cancels: Dict[str, Dict[str, Any]] = {}
        sandbox_recycled_request_ids: Dict[str, List[str]] = {}
        canceled_request_ids: Dict[str, List[str]] = {}
        target_toolbox_ids: set[str] = set()
        base = self._toolbox_runtime_base()
        for reg in target_regs:
            target_engine_id = str(dict(reg or {}).get("engine_id") or "").strip()
            if not target_engine_id:
                continue
            environment_key = self._toolbox_registration_environment_key(dict(reg or {}))
            pool = base.pool_registry.get(base.pool_key(environment_key)) if environment_key else None
            sibling_request_ids: List[str] = []
            target_request_status: Dict[str, Any] = {}
            if pool is not None and call_id:
                target_request_status = dict(pool.request_status(call_id).get("request") or {})
                for worker in list(pool.workers or []):
                    if str(worker.engine_id or "").strip() != target_engine_id:
                        continue
                    sibling_request_ids.extend(
                        [
                            str(active_request_id or "").strip()
                            for active_request_id in list(worker.active_request_ids or [])
                            if str(active_request_id or "").strip() and str(active_request_id or "").strip() != call_id
                        ]
                    )
            sibling_request_ids = sorted(set(sibling_request_ids))
            if environment_key and call_id:
                hosted_pool_cancels[target_engine_id] = dict(base.cancel_request(environment_key=environment_key, request_id=call_id))
                if str(hosted_pool_cancels[target_engine_id].get("status") or "") == "ok":
                    canceled_request_ids[target_engine_id] = [call_id]
            elif environment_key:
                canceled_requests: List[str] = []
                if pool is not None:
                    for worker in list(pool.workers or []):
                        if str(worker.engine_id or "").strip() != target_engine_id:
                            continue
                        for active_request_id in list(worker.active_request_ids or []):
                            cancel_out = dict(base.cancel_request(environment_key=environment_key, request_id=str(active_request_id or "")))
                            if str(cancel_out.get("status") or "") == "ok":
                                canceled_requests.append(str(active_request_id or ""))
                hosted_pool_cancels[target_engine_id] = {
                    "status": "ok" if canceled_requests else "not_found",
                    "environment_key": environment_key,
                    "canceled_request_ids": canceled_requests,
                }
                if canceled_requests:
                    canceled_request_ids[target_engine_id] = sorted(set(canceled_requests))
            reg_toolbox_id = self._registration_toolbox_id(dict(reg or {}))
            if reg_toolbox_id:
                target_toolbox_ids.add(reg_toolbox_id)
            queued_call_canceled = bool(
                call_id
                and str(target_request_status.get("status") or "") == "queued"
                and str(hosted_pool_cancels.get(target_engine_id, {}).get("status") or "") == "ok"
            )
            if queued_call_canceled:
                shutdown_out = {
                    "status": "ok",
                    "alive": False,
                    "skipped": True,
                    "reason": "queued_request_canceled",
                }
            else:
                shutdown_out = dict(self.shutdown(target_engine_id, timeout_seconds=float(timeout_seconds or 8.0)) or {})
            shutdown_results[target_engine_id] = shutdown_out
            if bool(shutdown_out.get("alive")) or str(shutdown_out.get("status") or "").strip() == "stop_failed":
                failed_engine_ids.append(target_engine_id)
            else:
                if not queued_call_canceled:
                    canceled_engine_ids.append(target_engine_id)
            if not queued_call_canceled and sibling_request_ids:
                recycled: List[str] = []
                for sibling_request_id in sibling_request_ids:
                    finished = base.finish_request(
                        environment_key=environment_key,
                        request_id=sibling_request_id,
                        status="error",
                        reason="sandbox_recycled",
                    )
                    if str(finished.get("status") or "") == "ok":
                        recycled.append(sibling_request_id)
                if recycled:
                    sandbox_recycled_request_ids[target_engine_id] = recycled

        repair_out: Dict[str, Any] = {}
        repaired_toolbox_ids: List[str] = []
        for target_toolbox_id in sorted(target_toolbox_ids):
            with self._locked_toolbox(target_toolbox_id):
                state = self._read_toolboxes()
                toolboxes = dict(state.get("toolboxes") or {})
                toolbox_row = dict(toolboxes.get(target_toolbox_id) or {})
                if toolbox_row:
                    toolbox_row = self._append_toolbox_cancel_event(
                        toolbox_row,
                        engine_ids=canceled_engine_ids,
                        tool_name=name or None,
                        tool_call_id=model_tool_call_id or None,
                        respawn=respawn,
                        non_restartable=self._toolbox_tool_non_restartable(toolbox_row, name),
                    )
                    toolboxes[target_toolbox_id] = toolbox_row
                    state["toolboxes"] = toolboxes
                    self._write_toolboxes(state)
                if respawn:
                    repair_piece = dict(
                        self.toolbox_repair(
                            toolbox_ids=[target_toolbox_id],
                            only_inconsistent=False,
                            details=False,
                        )
                        or {}
                    )
                    if repair_piece:
                        repaired_toolbox_ids.extend(
                            [
                                str(item or "").strip()
                                for item in list(repair_piece.get("repaired_toolbox_ids") or [])
                                if str(item or "").strip()
                            ]
                        )
                        if not repair_out:
                            repair_out = dict(repair_piece)
                        else:
                            repair_out.setdefault("repaired_toolbox_ids", [])
                            repair_out["repaired_toolbox_ids"] = sorted(
                                {
                                    *list(repair_out.get("repaired_toolbox_ids") or []),
                                    *list(repair_piece.get("repaired_toolbox_ids") or []),
                                }
                            )

        result = {
            "status": "ok",
            "engine_id": eid or None,
            "toolbox_id": tid or None,
            "tool_name": name or None,
            "tool_call_id": model_tool_call_id or None,
            "request_id": call_id or None,
            "respawn": bool(respawn),
            "outcome": (
                "canceled_and_repaired"
                if (canceled_engine_ids or canceled_request_ids) and repaired_toolbox_ids
                else "canceled"
                if canceled_engine_ids or canceled_request_ids
                else "noop"
                if not failed_engine_ids
                else "partial_failure"
            ),
            "canceled_engine_ids": sorted(canceled_engine_ids),
            "failed_engine_ids": sorted(failed_engine_ids),
            "canceled_request_ids": {key: sorted(value) for key, value in sorted(canceled_request_ids.items())},
            "sandbox_recycled_request_ids": {
                key: sorted(value) for key, value in sorted(sandbox_recycled_request_ids.items())
            },
            "repaired_toolbox_ids": sorted(repaired_toolbox_ids),
            "shutdown_results": shutdown_results,
            "hosted_pool_cancels": hosted_pool_cancels,
            "repair": repair_out,
        }
        if canceled_engine_ids or canceled_request_ids:
            return self._hosted_operations.finish(
                operation_id=operation_id,
                lifecycle=HostedOperationLifecycle.TERMINAL_CANCELLATION,
                envelope=result,
                reason=str(reason or "client_requested"),
            )
        failure_reason = "cancel_partial_failure" if failed_engine_ids else "cancel_target_not_active"
        return self._hosted_operations.finish(
            operation_id=operation_id,
            lifecycle=HostedOperationLifecycle.TERMINAL_FAILURE,
            envelope=result,
            reason=failure_reason,
        )

    def _wait_for_toolbox_executor_ready(
        self,
        engine_id: str,
        *,
        timeout_seconds: float = 8.0,
        poll_seconds: float = 0.1,
    ) -> Dict[str, Any]:
        eid = str(engine_id or "").strip()
        if not eid:
            raise ValueError("engine_id is required")
        try:
            reg = self._require_toolbox_executor_registration(eid, command_label="toolbox-ready")
        except Exception as exc:
            raise ToolboxRolloutError(
                f"toolbox_executor_missing:{eid}",
                code="toolbox_executor_missing",
                details={
                    "failure_phase": "spawned",
                    "engine_id": eid,
                    "reason": str(exc),
                },
            ) from exc
        deadline = time.time() + max(0.1, float(timeout_seconds or 8.0))
        last_error: Optional[Exception] = None
        while time.time() < deadline:
            try:
                desc = self.toolbox_describe(engine_id=eid, timeout_seconds=min(2.0, max(0.2, float(timeout_seconds or 8.0))))
                allowed = self._registration_allowed_tool_names(reg)
                reported = {
                    str(item or "").strip()
                    for item in list(
                        dict(desc or {}).get("all_registered_tool_names")
                        or []
                    )
                    if str(item or "").strip()
                }
                if allowed is not None and reported != allowed:
                    raise ToolboxRolloutError(
                        f"toolbox_executor_inventory_mismatch:{eid}",
                        code="toolbox_executor_inventory_mismatch",
                        details={
                            "failure_phase": "inventory_verified",
                            "engine_id": eid,
                            "expected_tool_names": sorted(allowed),
                            "actual_tool_names": sorted(reported),
                        },
                    )
                return desc
            except Exception as exc:
                last_error = exc
                time.sleep(max(0.05, float(poll_seconds or 0.1)))
        if isinstance(last_error, ToolboxRolloutError):
            raise ToolboxRolloutError(
                str(last_error),
                code=last_error.code,
                details=dict(last_error.details or {}),
            ) from last_error
        raise ToolboxRolloutError(
            f"toolbox_executor_not_ready:{eid}:{last_error}",
            code="toolbox_executor_not_ready",
            details={
                "failure_phase": "ready",
                "engine_id": eid,
                "timeout_seconds": float(timeout_seconds or 8.0),
                "reason": str(last_error or ""),
            },
        )

    def _ensure_toolbox_assignments_ready(
        self,
        assignments: List[Any],
        *,
        timeout_seconds: float = 8.0,
    ) -> Dict[str, Dict[str, Any]]:
        from ..toolbox_harness import ToolboxEnvironmentManager, ToolboxEnvironmentSpec

        ready: Dict[str, Dict[str, Any]] = {}
        environment_manager = ToolboxEnvironmentManager(self.hosting_root)
        for item in list(assignments or []):
            reg = dict(getattr(item, "registration", None) or {})
            engine_id = str(reg.get("engine_id") or "").strip()
            if not engine_id:
                continue
            started_at = time.time()
            try:
                desc = self._wait_for_toolbox_executor_ready(engine_id, timeout_seconds=timeout_seconds)
            except ToolboxRolloutError as exc:
                bundle = dict(reg.get("bundle") or {})
                details = dict(exc.details or {})
                details.setdefault("toolbox_id", str(bundle.get("toolbox_id") or getattr(item, "toolbox_id", "") or ""))
                details.setdefault(
                    "sandbox_profile_id",
                    str(bundle.get("sandbox_profile_id") or getattr(getattr(item, "sandbox_profile", None), "normalized_profile_id", lambda: "")() or ""),
                )
                details.setdefault("bundle_revision", str(bundle.get("bundle_revision") or ""))
                details.setdefault("engine_id", engine_id)
                raise ToolboxRolloutError(str(exc), code=exc.code, details=details) from exc
            ready_at = time.time()
            tool_names = [
                str(name or "").strip()
                for name in list(
                    dict(desc or {}).get("all_registered_tool_names")
                    or []
                )
                if str(name or "").strip()
            ]
            environment = dict(reg.get("environment") or {})
            receipt_verification_status = None
            install_execution_status = None
            if str(reg.get("routing_state") or "") == "candidate":
                expected_names = [
                    str(name or "").strip()
                    for name in list(dict(reg.get("tool_access") or {}).get("allowed_tool_names") or [])
                    if str(name or "").strip()
                ]
                if len(tool_names) != len(set(tool_names)) or set(tool_names) != set(expected_names):
                    raise ToolboxRolloutError(
                        f"toolbox inventory mismatch for {engine_id}",
                        code="toolbox_candidate_inventory_mismatch",
                        details={"engine_id": engine_id, "failure_phase": "inventory"},
                    )
                bundle = dict(reg.get("bundle") or {})
                profile = getattr(item, "resolved_profile", None)
                expected_profile_id = str(getattr(profile, "profile_id", "") or "")
                expected_environment_key = str(getattr(profile, "environment_key", "") or "")
                if (
                    not expected_profile_id
                    or bundle.get("resolved_profile_id") != expected_profile_id
                    or environment.get("environment_key") != expected_environment_key
                    or environment.get("verification_state") != "verified"
                    or environment.get("verification_receipt_contract")
                    != "hosting.toolbox.hermetic_environment_receipt.v1"
                ):
                    raise ToolboxRolloutError(
                        f"toolbox candidate metadata mismatch for {engine_id}",
                        code="toolbox_candidate_metadata_mismatch",
                        details={"engine_id": engine_id, "failure_phase": "metadata"},
                    )
                receipt_path = Path(str(environment.get("venv_path") or "")) / "verification-receipt.json"
                try:
                    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
                except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                    raise ToolboxRolloutError(
                        f"toolbox environment receipt unavailable for {engine_id}",
                        code="toolbox_environment_receipt_unverified",
                        details={"engine_id": engine_id, "failure_phase": "environment_receipt"},
                    ) from exc
                if (
                    not isinstance(receipt, dict)
                    or receipt.get("contract") != "hosting.toolbox.hermetic_environment_receipt.v1"
                    or receipt.get("state") != "verified"
                    or receipt.get("environment_key") != expected_environment_key
                ):
                    raise ToolboxRolloutError(
                        f"toolbox environment receipt mismatch for {engine_id}",
                        code="toolbox_environment_receipt_unverified",
                        details={"engine_id": engine_id, "failure_phase": "environment_receipt"},
                    )
                receipt_verification_status = "ok"
                install_execution_status = "ok"
            elif environment:
                spec = ToolboxEnvironmentSpec.from_dict(environment)
                metadata = environment_manager.read_environment_metadata(spec)
                install_execution_status = str(dict(metadata.get("install_execution") or {}).get("status") or "").strip() or None
                receipt_verification_status = str(
                    dict(metadata.get("install_receipt_verification") or {}).get("status") or ""
                ).strip() or None
                if install_execution_status == "ok" and receipt_verification_status != "ok":
                    raise ToolboxRolloutError(
                        f"environment receipt verification not ready for {engine_id}",
                        code="toolbox_environment_receipt_unverified",
                        details={
                            "engine_id": engine_id,
                            "install_execution_status": install_execution_status,
                            "install_receipt_verification_status": receipt_verification_status,
                            "toolbox_id": str(dict(reg.get("bundle") or {}).get("toolbox_id") or getattr(item, "toolbox_id", "") or ""),
                            "sandbox_profile_id": str(
                                dict(reg.get("bundle") or {}).get("sandbox_profile_id")
                                or getattr(getattr(item, "sandbox_profile", None), "normalized_profile_id", lambda: "")()
                                or ""
                            ),
                        },
                    )
            ready[engine_id] = {
                "engine_id": engine_id,
                "ready": True,
                "ready_at": ready_at,
                "warmup_ms": int(max(0.0, (ready_at - started_at) * 1000.0)),
                "tool_inventory_ok": True,
                "tool_count": len(tool_names),
                "all_registered_tool_names": tool_names,
                "install_execution_status": install_execution_status,
                "install_receipt_verification_status": receipt_verification_status,
            }
        return ready

    def toolbox_register_auto(
        self,
        *,
        toolbox_id: str,
        requests: List[Dict[str, Any]],
        python_executable: Optional[str] = None,
        worker_profile_class: str = "generic",
    ) -> Dict[str, Any]:
        tid = str(toolbox_id or "").strip()
        if not tid:
            raise ValueError("toolbox_id is required")
        return self._run_locked_toolbox_call(
            tid,
            self._toolbox_register_auto_impl,
            toolbox_id=tid,
            requests=list(requests or []),
            python_executable=python_executable,
            worker_profile_class=worker_profile_class,
        )

    def _toolbox_register_auto_impl(
        self,
        *,
        toolbox_id: str,
        requests: List[Dict[str, Any]],
        python_executable: Optional[str] = None,
        worker_profile_class: str = "generic",
    ) -> Dict[str, Any]:
        from ..toolbox_harness import (
            ToolboxAutoAssignmentRequest,
            SandboxProfileSpec,
            ToolboxBundleStager,
            ToolboxManualAssignmentRequest,
            ToolboxSandboxOrchestrator,
        )

        tid = str(toolbox_id or "").strip()
        if not tid:
            raise ValueError("toolbox_id is required")
        if not list(requests or []):
            raise ValueError("requests are required")
        merged_requests, state, toolboxes = self._merge_toolbox_auto_requests(
            toolbox_id=tid,
            incoming_requests=list(requests or []),
        )
        toolbox_row_existing = dict(toolboxes.get(tid) or {})
        manual_requests = [
            ToolboxManualAssignmentRequest.from_runtime_dict(dict(item or {}))
            for item in list(toolbox_row_existing.get("manual_requests") or [])
        ]
        intrinsics_row = dict(toolbox_row_existing.get("intrinsics") or {})
        intrinsic_names = self._normalize_intrinsic_tool_names(
            [str(item or "").strip() for item in list(intrinsics_row.get("names") or []) if str(item or "").strip()],
            include_guides=bool(intrinsics_row.get("with_intrinsic_guides", False)),
        )
        intrinsic_profile = SandboxProfileSpec.from_dict(dict(intrinsics_row.get("sandbox_profile") or {})) if intrinsic_names else None
        with_intrinsic_guides = bool(intrinsics_row.get("with_intrinsic_guides", False))
        existing_profiles = dict(dict(toolboxes.get(tid) or {}).get("profiles") or {})
        runtime = self._toolbox_runtime_defaults(
            toolbox_row_existing,
            python_executable=python_executable,
            worker_profile_class=worker_profile_class,
        )

        old_regs_by_profile: Dict[str, str] = {}
        for reg in self._toolbox_executor_registrations(tid):
            old_regs_by_profile[self._registration_sandbox_profile_id(reg)] = str(reg.get("engine_id") or "").strip()

        orchestrator = ToolboxSandboxOrchestrator(
            service=self,
            stager=ToolboxBundleStager(self.hosting_root),
            python_executable=runtime.get("python_executable"),
        )
        assignments = orchestrator.spawn_assignments(
            toolbox_id=tid,
            requests=merged_requests,
            manual_requests=manual_requests,
            intrinsic_tool_names=intrinsic_names,
            intrinsic_profile=intrinsic_profile,
            with_intrinsic_guides=with_intrinsic_guides,
            worker_profile_class=str(runtime.get("worker_profile_class") or "generic"),
        )
        try:
            ready_rollout = self._ensure_toolbox_assignments_ready(assignments, timeout_seconds=8.0)
        except Exception:
            for item in assignments:
                reg = dict(item.registration or {})
                engine_id = str(reg.get("engine_id") or "").strip()
                if engine_id:
                    self._retire_toolbox_registration(engine_id)
            self._cleanup_unused_toolbox_environments(state)
            raise

        new_profiles: Dict[str, Dict[str, Any]] = {}
        spawned_engine_ids: List[str] = []
        replaced_engine_ids: List[str] = []
        for item in assignments:
            profile_id = item.sandbox_profile.normalized_profile_id()
            reg = dict(item.registration or {})
            engine_id = str(reg.get("engine_id") or "").strip()
            if engine_id:
                spawned_engine_ids.append(engine_id)
            old_engine_id = str(old_regs_by_profile.get(profile_id) or "").strip()
            bundle_revision = str(dict(reg.get("bundle") or {}).get("bundle_revision") or "")
            if old_engine_id and old_engine_id != engine_id:
                replaced_engine_ids.append(old_engine_id)
            profile_requests = [
                req.to_runtime_dict()
                for req in merged_requests
                if req.sandbox_profile.normalized_profile_id() == profile_id
            ]
            new_profiles[profile_id] = {
                "sandbox_profile": item.sandbox_profile.to_dict(),
                "requests": profile_requests,
                "engine_id": engine_id,
                "bundle_revision": bundle_revision,
                "environment": dict(reg.get("environment") or {}),
                "rollout": dict(ready_rollout.get(engine_id) or {}),
                "rollout_history": self._append_toolbox_rollout_history(
                    dict(existing_profiles.get(profile_id) or {}),
                    rollout=dict(ready_rollout.get(engine_id) or {}),
                    action="register_auto",
                    engine_id=engine_id,
                    bundle_revision=bundle_revision,
                    replaced_engine_id=old_engine_id,
                ),
            }

        for profile_id, old_engine_id in old_regs_by_profile.items():
            if profile_id not in new_profiles and old_engine_id:
                replaced_engine_ids.append(old_engine_id)

        for old_engine_id in replaced_engine_ids:
            self._retire_toolbox_registration(old_engine_id)

        toolbox_row = {
            "toolbox_id": tid,
            "requests": [req.to_runtime_dict() for req in merged_requests],
            "manual_requests": [req.to_runtime_dict() for req in manual_requests],
            "profiles": new_profiles,
            "runtime": runtime,
        }
        if intrinsic_names:
            toolbox_row["intrinsics"] = {
                "names": intrinsic_names,
                "sandbox_profile": (intrinsic_profile or SandboxProfileSpec(profile_id="default")).to_dict(),
                "with_intrinsic_guides": with_intrinsic_guides,
            }
        toolboxes[tid] = toolbox_row
        state["toolboxes"] = toolboxes
        self._write_toolboxes(state)
        removed_environment_keys = self._cleanup_unused_toolbox_environments(state)
        return {
            "status": "ok",
            "toolbox_id": tid,
            "request_count": len(merged_requests),
            "profiles": new_profiles,
            "spawned_engine_ids": spawned_engine_ids,
            "ready_engine_ids": list(ready_rollout.keys()),
            "rollout": ready_rollout,
            "replaced_engine_ids": replaced_engine_ids,
            "removed_environment_keys": removed_environment_keys,
        }

    def toolbox_unregister_auto(
        self,
        *,
        toolbox_id: str,
        tool_keys: List[str],
        python_executable: Optional[str] = None,
        worker_profile_class: str = "generic",
    ) -> Dict[str, Any]:
        tid = str(toolbox_id or "").strip()
        if not tid:
            raise ValueError("toolbox_id is required")
        return self._run_locked_toolbox_call(
            tid,
            self._toolbox_unregister_auto_impl,
            toolbox_id=tid,
            tool_keys=list(tool_keys or []),
            python_executable=python_executable,
            worker_profile_class=worker_profile_class,
        )

    def _toolbox_unregister_auto_impl(
        self,
        *,
        toolbox_id: str,
        tool_keys: List[str],
        python_executable: Optional[str] = None,
        worker_profile_class: str = "generic",
    ) -> Dict[str, Any]:
        from ..toolbox_harness import (
            SandboxProfileSpec,
            ToolboxBundleStager,
            ToolboxManualAssignmentRequest,
            ToolboxSandboxOrchestrator,
        )

        tid = str(toolbox_id or "").strip()
        if not tid:
            raise ValueError("toolbox_id is required")
        keys = [str(item or "").strip() for item in list(tool_keys or []) if str(item or "").strip()]
        if not keys:
            raise ValueError("tool_keys are required")

        merged_requests, state, toolboxes = self._merge_toolbox_auto_requests(
            toolbox_id=tid,
            remove_keys=keys,
        )
        toolbox_row_existing = dict(toolboxes.get(tid) or {})
        manual_requests = [
            ToolboxManualAssignmentRequest.from_runtime_dict(dict(item or {}))
            for item in list(toolbox_row_existing.get("manual_requests") or [])
        ]
        intrinsics_row = dict(toolbox_row_existing.get("intrinsics") or {})
        intrinsic_names = self._normalize_intrinsic_tool_names(
            [str(item or "").strip() for item in list(intrinsics_row.get("names") or []) if str(item or "").strip()],
            include_guides=bool(intrinsics_row.get("with_intrinsic_guides", False)),
        )
        intrinsic_profile = SandboxProfileSpec.from_dict(dict(intrinsics_row.get("sandbox_profile") or {})) if intrinsic_names else None
        with_intrinsic_guides = bool(intrinsics_row.get("with_intrinsic_guides", False))
        existing_profiles = dict(dict(toolboxes.get(tid) or {}).get("profiles") or {})
        runtime = self._toolbox_runtime_defaults(
            toolbox_row_existing,
            python_executable=python_executable,
            worker_profile_class=worker_profile_class,
        )
        old_regs_by_profile: Dict[str, str] = {}
        for reg in self._toolbox_executor_registrations(tid):
            old_regs_by_profile[self._registration_sandbox_profile_id(reg)] = str(reg.get("engine_id") or "").strip()

        replaced_engine_ids: List[str] = []
        spawned_engine_ids: List[str] = []
        new_profiles: Dict[str, Dict[str, Any]] = {}

        if merged_requests or manual_requests or intrinsic_names:
            orchestrator = ToolboxSandboxOrchestrator(
                service=self,
                stager=ToolboxBundleStager(self.hosting_root),
                python_executable=runtime.get("python_executable"),
            )
            assignments = orchestrator.spawn_assignments(
                toolbox_id=tid,
                requests=merged_requests,
                manual_requests=manual_requests,
                intrinsic_tool_names=intrinsic_names,
                intrinsic_profile=intrinsic_profile,
                with_intrinsic_guides=with_intrinsic_guides,
                worker_profile_class=str(runtime.get("worker_profile_class") or "generic"),
            )
            try:
                ready_rollout = self._ensure_toolbox_assignments_ready(assignments, timeout_seconds=8.0)
            except Exception:
                for item in assignments:
                    reg = dict(item.registration or {})
                    engine_id = str(reg.get("engine_id") or "").strip()
                    if engine_id:
                        self._retire_toolbox_registration(engine_id)
                self._cleanup_unused_toolbox_environments(state)
                raise
            for item in assignments:
                profile_id = item.sandbox_profile.normalized_profile_id()
                reg = dict(item.registration or {})
                engine_id = str(reg.get("engine_id") or "").strip()
                if engine_id:
                    spawned_engine_ids.append(engine_id)
                old_engine_id = str(old_regs_by_profile.get(profile_id) or "").strip()
                bundle_revision = str(dict(reg.get("bundle") or {}).get("bundle_revision") or "")
                if old_engine_id and old_engine_id != engine_id:
                    replaced_engine_ids.append(old_engine_id)
                profile_requests = [
                    req.to_runtime_dict()
                    for req in merged_requests
                    if req.sandbox_profile.normalized_profile_id() == profile_id
                ]
                new_profiles[profile_id] = {
                    "sandbox_profile": item.sandbox_profile.to_dict(),
                    "requests": profile_requests,
                    "engine_id": engine_id,
                    "bundle_revision": bundle_revision,
                    "environment": dict(reg.get("environment") or {}),
                    "rollout": dict(ready_rollout.get(engine_id) or {}),
                    "rollout_history": self._append_toolbox_rollout_history(
                        dict(existing_profiles.get(profile_id) or {}),
                        rollout=dict(ready_rollout.get(engine_id) or {}),
                        action="unregister_auto",
                        engine_id=engine_id,
                        bundle_revision=bundle_revision,
                        replaced_engine_id=old_engine_id,
                    ),
                }
            for profile_id, old_engine_id in old_regs_by_profile.items():
                if profile_id not in new_profiles and old_engine_id:
                    replaced_engine_ids.append(old_engine_id)
            for old_engine_id in replaced_engine_ids:
                self._retire_toolbox_registration(old_engine_id)
            toolboxes[tid] = {
                "toolbox_id": tid,
                "requests": [req.to_runtime_dict() for req in merged_requests],
                "manual_requests": [req.to_runtime_dict() for req in manual_requests],
                "profiles": new_profiles,
                "runtime": runtime,
                **(
                    {
                        "intrinsics": {
                            "names": intrinsic_names,
                            "sandbox_profile": (intrinsic_profile or SandboxProfileSpec(profile_id="default")).to_dict(),
                            "with_intrinsic_guides": with_intrinsic_guides,
                        }
                    }
                    if intrinsic_names
                    else {}
                ),
            }
        else:
            ready_rollout = {}
            for old_engine_id in old_regs_by_profile.values():
                if old_engine_id:
                    replaced_engine_ids.append(old_engine_id)
            for old_engine_id in replaced_engine_ids:
                self._retire_toolbox_registration(old_engine_id)
            toolboxes.pop(tid, None)

        state["toolboxes"] = toolboxes
        self._write_toolboxes(state)
        removed_environment_keys = self._cleanup_unused_toolbox_environments(state)
        return {
            "status": "ok",
            "toolbox_id": tid,
            "remaining_request_count": len(merged_requests),
            "profiles": new_profiles,
            "spawned_engine_ids": spawned_engine_ids,
            "ready_engine_ids": list(ready_rollout.keys()),
            "rollout": ready_rollout,
            "replaced_engine_ids": replaced_engine_ids,
            "removed_tool_keys": keys,
            "toolbox_removed": not merged_requests and not manual_requests and not intrinsic_names,
            "removed_environment_keys": removed_environment_keys,
        }

    def toolbox_register_intrinsics(
        self,
        *,
        toolbox_id: str,
        intrinsic_tool_names: List[str],
        include_guides: bool = False,
        sandbox_profile: Optional[Dict[str, Any]] = None,
        python_executable: Optional[str] = None,
        worker_profile_class: str = "generic",
    ) -> Dict[str, Any]:
        tid = str(toolbox_id or "").strip()
        if not tid:
            raise ValueError("toolbox_id is required")
        return self._run_locked_toolbox_call(
            tid,
            self._toolbox_register_intrinsics_impl,
            toolbox_id=tid,
            intrinsic_tool_names=list(intrinsic_tool_names or []),
            include_guides=include_guides,
            sandbox_profile=dict(sandbox_profile or {}) if isinstance(sandbox_profile, dict) else None,
            python_executable=python_executable,
            worker_profile_class=worker_profile_class,
        )

    def _toolbox_register_intrinsics_impl(
        self,
        *,
        toolbox_id: str,
        intrinsic_tool_names: List[str],
        include_guides: bool = False,
        sandbox_profile: Optional[Dict[str, Any]] = None,
        python_executable: Optional[str] = None,
        worker_profile_class: str = "generic",
    ) -> Dict[str, Any]:
        from ..toolbox_harness import SandboxProfileSpec, ToolboxBundleStager, ToolboxSandboxOrchestrator

        tid = str(toolbox_id or "").strip()
        if not tid:
            raise ValueError("toolbox_id is required")
        names = self._normalize_intrinsic_tool_names(
            [str(item or "").strip() for item in list(intrinsic_tool_names or []) if str(item or "").strip()],
            include_guides=bool(include_guides),
        )
        if not names:
            raise ValueError("intrinsic_tool_names are required")
        merged_requests, state, toolboxes = self._merge_toolbox_auto_requests(toolbox_id=tid)
        toolbox_row_existing = dict(toolboxes.get(tid) or {})
        existing_profiles = dict(toolbox_row_existing.get("profiles") or {})
        runtime = self._toolbox_runtime_defaults(
            toolbox_row_existing,
            python_executable=python_executable,
            worker_profile_class=worker_profile_class,
        )
        existing_intrinsics = dict(toolbox_row_existing.get("intrinsics") or {})
        existing_names = self._normalize_intrinsic_tool_names(
            [str(item or "").strip() for item in list(existing_intrinsics.get("names") or []) if str(item or "").strip()],
            include_guides=bool(existing_intrinsics.get("with_intrinsic_guides", False)),
        )
        merged_names = self._normalize_intrinsic_tool_names(existing_names + names, include_guides=False)
        intrinsic_profile = SandboxProfileSpec.from_dict(
            dict(sandbox_profile or existing_intrinsics.get("sandbox_profile") or {})
        )
        with_intrinsic_guides = bool(include_guides or existing_intrinsics.get("with_intrinsic_guides", False))

        old_regs_by_profile: Dict[str, str] = {}
        for reg in self._toolbox_executor_registrations(tid):
            old_regs_by_profile[self._registration_sandbox_profile_id(reg)] = str(reg.get("engine_id") or "").strip()

        orchestrator = ToolboxSandboxOrchestrator(
            service=self,
            stager=ToolboxBundleStager(self.hosting_root),
            python_executable=runtime.get("python_executable"),
        )
        assignments = orchestrator.spawn_assignments(
            toolbox_id=tid,
            requests=merged_requests,
            intrinsic_tool_names=merged_names,
            intrinsic_profile=intrinsic_profile,
            with_intrinsic_guides=with_intrinsic_guides,
            worker_profile_class=str(runtime.get("worker_profile_class") or "generic"),
        )
        try:
            ready_rollout = self._ensure_toolbox_assignments_ready(assignments, timeout_seconds=8.0)
        except Exception:
            for item in assignments:
                reg = dict(item.registration or {})
                engine_id = str(reg.get("engine_id") or "").strip()
                if engine_id:
                    self._retire_toolbox_registration(engine_id)
            self._cleanup_unused_toolbox_environments(state)
            raise

        new_profiles: Dict[str, Dict[str, Any]] = {}
        spawned_engine_ids: List[str] = []
        replaced_engine_ids: List[str] = []
        for item in assignments:
            profile_id = item.sandbox_profile.normalized_profile_id()
            reg = dict(item.registration or {})
            engine_id = str(reg.get("engine_id") or "").strip()
            if engine_id:
                spawned_engine_ids.append(engine_id)
            old_engine_id = str(old_regs_by_profile.get(profile_id) or "").strip()
            bundle_revision = str(dict(reg.get("bundle") or {}).get("bundle_revision") or "")
            if old_engine_id and old_engine_id != engine_id:
                replaced_engine_ids.append(old_engine_id)
            profile_requests = [
                req.to_runtime_dict()
                for req in merged_requests
                if req.sandbox_profile.normalized_profile_id() == profile_id
            ]
            new_profiles[profile_id] = {
                "sandbox_profile": item.sandbox_profile.to_dict(),
                "requests": profile_requests,
                "engine_id": engine_id,
                "bundle_revision": bundle_revision,
                "environment": dict(reg.get("environment") or {}),
                "rollout": dict(ready_rollout.get(engine_id) or {}),
                "rollout_history": self._append_toolbox_rollout_history(
                    dict(existing_profiles.get(profile_id) or {}),
                    rollout=dict(ready_rollout.get(engine_id) or {}),
                    action="register_intrinsics",
                    engine_id=engine_id,
                    bundle_revision=bundle_revision,
                    replaced_engine_id=old_engine_id,
                ),
            }

        for profile_id, old_engine_id in old_regs_by_profile.items():
            if profile_id not in new_profiles and old_engine_id:
                replaced_engine_ids.append(old_engine_id)
        for old_engine_id in replaced_engine_ids:
            self._retire_toolbox_registration(old_engine_id)

        toolboxes[tid] = {
            "toolbox_id": tid,
            "requests": [req.to_runtime_dict() for req in merged_requests],
            "profiles": new_profiles,
            "runtime": runtime,
            "intrinsics": {
                "names": merged_names,
                "sandbox_profile": intrinsic_profile.to_dict(),
                "with_intrinsic_guides": with_intrinsic_guides,
            },
        }
        state["toolboxes"] = toolboxes
        self._write_toolboxes(state)
        removed_environment_keys = self._cleanup_unused_toolbox_environments(state)
        return {
            "status": "ok",
            "toolbox_id": tid,
            "intrinsic_tool_names": merged_names,
            "profiles": new_profiles,
            "spawned_engine_ids": spawned_engine_ids,
            "ready_engine_ids": list(ready_rollout.keys()),
            "rollout": ready_rollout,
            "replaced_engine_ids": replaced_engine_ids,
            "removed_environment_keys": removed_environment_keys,
        }

    def toolbox_unregister_intrinsics(
        self,
        *,
        toolbox_id: str,
        intrinsic_tool_names: List[str],
        include_guides: bool = False,
        python_executable: Optional[str] = None,
        worker_profile_class: str = "generic",
    ) -> Dict[str, Any]:
        tid = str(toolbox_id or "").strip()
        if not tid:
            raise ValueError("toolbox_id is required")
        return self._run_locked_toolbox_call(
            tid,
            self._toolbox_unregister_intrinsics_impl,
            toolbox_id=tid,
            intrinsic_tool_names=list(intrinsic_tool_names or []),
            include_guides=include_guides,
            python_executable=python_executable,
            worker_profile_class=worker_profile_class,
        )

    def _toolbox_unregister_intrinsics_impl(
        self,
        *,
        toolbox_id: str,
        intrinsic_tool_names: List[str],
        include_guides: bool = False,
        python_executable: Optional[str] = None,
        worker_profile_class: str = "generic",
    ) -> Dict[str, Any]:
        from ..toolbox_harness import SandboxProfileSpec, ToolboxBundleStager, ToolboxSandboxOrchestrator

        tid = str(toolbox_id or "").strip()
        if not tid:
            raise ValueError("toolbox_id is required")
        remove_names = self._normalize_intrinsic_tool_names(
            [str(item or "").strip() for item in list(intrinsic_tool_names or []) if str(item or "").strip()],
            include_guides=bool(include_guides),
        )
        if not remove_names:
            raise ValueError("intrinsic_tool_names are required")
        merged_requests, state, toolboxes = self._merge_toolbox_auto_requests(toolbox_id=tid)
        toolbox_row_existing = dict(toolboxes.get(tid) or {})
        existing_profiles = dict(toolbox_row_existing.get("profiles") or {})
        runtime = self._toolbox_runtime_defaults(
            toolbox_row_existing,
            python_executable=python_executable,
            worker_profile_class=worker_profile_class,
        )
        existing_intrinsics = dict(toolbox_row_existing.get("intrinsics") or {})
        current_names = self._normalize_intrinsic_tool_names(
            [str(item or "").strip() for item in list(existing_intrinsics.get("names") or []) if str(item or "").strip()],
            include_guides=bool(existing_intrinsics.get("with_intrinsic_guides", False)),
        )
        remaining_names = [name for name in current_names if name not in set(remove_names)]
        intrinsic_profile = SandboxProfileSpec.from_dict(dict(existing_intrinsics.get("sandbox_profile") or {}))
        with_intrinsic_guides = bool(existing_intrinsics.get("with_intrinsic_guides", False))

        old_regs_by_profile: Dict[str, str] = {}
        for reg in self._toolbox_executor_registrations(tid):
            old_regs_by_profile[self._registration_sandbox_profile_id(reg)] = str(reg.get("engine_id") or "").strip()

        replaced_engine_ids: List[str] = []
        spawned_engine_ids: List[str] = []
        new_profiles: Dict[str, Dict[str, Any]] = {}
        if merged_requests or remaining_names:
            orchestrator = ToolboxSandboxOrchestrator(
                service=self,
                stager=ToolboxBundleStager(self.hosting_root),
                python_executable=runtime.get("python_executable"),
            )
            assignments = orchestrator.spawn_assignments(
                toolbox_id=tid,
                requests=merged_requests,
                intrinsic_tool_names=remaining_names,
                intrinsic_profile=intrinsic_profile if remaining_names else None,
                with_intrinsic_guides=with_intrinsic_guides,
                worker_profile_class=str(runtime.get("worker_profile_class") or "generic"),
            )
            try:
                ready_rollout = self._ensure_toolbox_assignments_ready(assignments, timeout_seconds=8.0)
            except Exception:
                for item in assignments:
                    reg = dict(item.registration or {})
                    engine_id = str(reg.get("engine_id") or "").strip()
                    if engine_id:
                        self._retire_toolbox_registration(engine_id)
                self._cleanup_unused_toolbox_environments(state)
                raise
            for item in assignments:
                profile_id = item.sandbox_profile.normalized_profile_id()
                reg = dict(item.registration or {})
                engine_id = str(reg.get("engine_id") or "").strip()
                if engine_id:
                    spawned_engine_ids.append(engine_id)
                old_engine_id = str(old_regs_by_profile.get(profile_id) or "").strip()
                bundle_revision = str(dict(reg.get("bundle") or {}).get("bundle_revision") or "")
                if old_engine_id and old_engine_id != engine_id:
                    replaced_engine_ids.append(old_engine_id)
                profile_requests = [
                    req.to_runtime_dict()
                    for req in merged_requests
                    if req.sandbox_profile.normalized_profile_id() == profile_id
                ]
                new_profiles[profile_id] = {
                    "sandbox_profile": item.sandbox_profile.to_dict(),
                    "requests": profile_requests,
                    "engine_id": engine_id,
                    "bundle_revision": bundle_revision,
                    "environment": dict(reg.get("environment") or {}),
                    "rollout": dict(ready_rollout.get(engine_id) or {}),
                    "rollout_history": self._append_toolbox_rollout_history(
                        dict(existing_profiles.get(profile_id) or {}),
                        rollout=dict(ready_rollout.get(engine_id) or {}),
                        action="unregister_intrinsics",
                        engine_id=engine_id,
                        bundle_revision=bundle_revision,
                        replaced_engine_id=old_engine_id,
                    ),
                }
            for profile_id, old_engine_id in old_regs_by_profile.items():
                if profile_id not in new_profiles and old_engine_id:
                    replaced_engine_ids.append(old_engine_id)
            for old_engine_id in replaced_engine_ids:
                self._retire_toolbox_registration(old_engine_id)
            toolboxes[tid] = {
                "toolbox_id": tid,
                "requests": [req.to_runtime_dict() for req in merged_requests],
                "profiles": new_profiles,
                "runtime": runtime,
                **(
                    {
                        "intrinsics": {
                            "names": remaining_names,
                            "sandbox_profile": intrinsic_profile.to_dict(),
                            "with_intrinsic_guides": with_intrinsic_guides,
                        }
                    }
                    if remaining_names
                    else {}
                ),
            }
        else:
            ready_rollout = {}
            for old_engine_id in old_regs_by_profile.values():
                if old_engine_id:
                    replaced_engine_ids.append(old_engine_id)
            for old_engine_id in replaced_engine_ids:
                self._retire_toolbox_registration(old_engine_id)
            toolboxes.pop(tid, None)
        state["toolboxes"] = toolboxes
        self._write_toolboxes(state)
        removed_environment_keys = self._cleanup_unused_toolbox_environments(state)
        return {
            "status": "ok",
            "toolbox_id": tid,
            "remaining_intrinsic_tool_names": remaining_names,
            "profiles": new_profiles,
            "spawned_engine_ids": spawned_engine_ids,
            "ready_engine_ids": list(ready_rollout.keys()),
            "rollout": ready_rollout,
            "replaced_engine_ids": replaced_engine_ids,
            "toolbox_removed": not merged_requests and not remaining_names,
            "removed_environment_keys": removed_environment_keys,
        }

    def toolbox_register_manual(
        self,
        *,
        toolbox_id: str,
        requests: List[Dict[str, Any]],
        python_executable: Optional[str] = None,
        worker_profile_class: str = "generic",
    ) -> Dict[str, Any]:
        tid = str(toolbox_id or "").strip()
        if not tid:
            raise ValueError("toolbox_id is required")
        return self._run_locked_toolbox_call(
            tid,
            self._toolbox_register_manual_impl,
            toolbox_id=tid,
            requests=list(requests or []),
            python_executable=python_executable,
            worker_profile_class=worker_profile_class,
        )

    def _toolbox_register_manual_impl(
        self,
        *,
        toolbox_id: str,
        requests: List[Dict[str, Any]],
        python_executable: Optional[str] = None,
        worker_profile_class: str = "generic",
    ) -> Dict[str, Any]:
        from ..toolbox_harness import (
            SandboxProfileSpec,
            ToolboxBundleStager,
            ToolboxManualAssignmentRequest,
            ToolboxSandboxOrchestrator,
        )

        tid = str(toolbox_id or "").strip()
        if not tid:
            raise ValueError("toolbox_id is required")
        if not list(requests or []):
            raise ValueError("requests are required")
        manual_requests, state, toolboxes = self._merge_toolbox_manual_requests(
            toolbox_id=tid,
            incoming_requests=list(requests or []),
        )
        toolbox_row_existing = dict(toolboxes.get(tid) or {})
        auto_requests = [
            ToolboxAutoAssignmentRequest.from_runtime_dict(dict(item or {}))
            for item in list(toolbox_row_existing.get("requests") or [])
        ]
        intrinsics_row = dict(toolbox_row_existing.get("intrinsics") or {})
        intrinsic_names = self._normalize_intrinsic_tool_names(
            [str(item or "").strip() for item in list(intrinsics_row.get("names") or []) if str(item or "").strip()],
            include_guides=bool(intrinsics_row.get("with_intrinsic_guides", False)),
        )
        intrinsic_profile = SandboxProfileSpec.from_dict(dict(intrinsics_row.get("sandbox_profile") or {})) if intrinsic_names else None
        with_intrinsic_guides = bool(intrinsics_row.get("with_intrinsic_guides", False))
        existing_profiles = dict(dict(toolboxes.get(tid) or {}).get("profiles") or {})
        runtime = self._toolbox_runtime_defaults(
            toolbox_row_existing,
            python_executable=python_executable,
            worker_profile_class=worker_profile_class,
        )
        old_regs_by_profile: Dict[str, str] = {}
        for reg in self._toolbox_executor_registrations(tid):
            old_regs_by_profile[self._registration_sandbox_profile_id(reg)] = str(reg.get("engine_id") or "").strip()

        orchestrator = ToolboxSandboxOrchestrator(
            service=self,
            stager=ToolboxBundleStager(self.hosting_root),
            python_executable=runtime.get("python_executable"),
        )
        assignments = orchestrator.spawn_assignments(
            toolbox_id=tid,
            requests=auto_requests,
            manual_requests=manual_requests,
            intrinsic_tool_names=intrinsic_names,
            intrinsic_profile=intrinsic_profile,
            with_intrinsic_guides=with_intrinsic_guides,
            worker_profile_class=str(runtime.get("worker_profile_class") or "generic"),
        )
        try:
            ready_rollout = self._ensure_toolbox_assignments_ready(assignments, timeout_seconds=8.0)
        except Exception:
            for item in assignments:
                reg = dict(item.registration or {})
                engine_id = str(reg.get("engine_id") or "").strip()
                if engine_id:
                    self._retire_toolbox_registration(engine_id)
            self._cleanup_unused_toolbox_environments(state)
            raise

        new_profiles: Dict[str, Dict[str, Any]] = {}
        spawned_engine_ids: List[str] = []
        replaced_engine_ids: List[str] = []
        for item in assignments:
            profile_id = item.sandbox_profile.normalized_profile_id()
            reg = dict(item.registration or {})
            engine_id = str(reg.get("engine_id") or "").strip()
            if engine_id:
                spawned_engine_ids.append(engine_id)
            old_engine_id = str(old_regs_by_profile.get(profile_id) or "").strip()
            bundle_revision = str(dict(reg.get("bundle") or {}).get("bundle_revision") or "")
            if old_engine_id and old_engine_id != engine_id:
                replaced_engine_ids.append(old_engine_id)
            profile_auto_requests = [
                req.to_runtime_dict()
                for req in auto_requests
                if req.sandbox_profile.normalized_profile_id() == profile_id
            ]
            profile_manual_requests = [
                req.to_runtime_dict()
                for req in manual_requests
                if req.sandbox_profile.normalized_profile_id() == profile_id
            ]
            new_profiles[profile_id] = {
                "sandbox_profile": item.sandbox_profile.to_dict(),
                "requests": profile_auto_requests,
                "manual_requests": profile_manual_requests,
                "engine_id": engine_id,
                "bundle_revision": bundle_revision,
                "environment": dict(reg.get("environment") or {}),
                "rollout": dict(ready_rollout.get(engine_id) or {}),
                "rollout_history": self._append_toolbox_rollout_history(
                    dict(existing_profiles.get(profile_id) or {}),
                    rollout=dict(ready_rollout.get(engine_id) or {}),
                    action="register_manual",
                    engine_id=engine_id,
                    bundle_revision=bundle_revision,
                    replaced_engine_id=old_engine_id,
                ),
            }

        for profile_id, old_engine_id in old_regs_by_profile.items():
            if profile_id not in new_profiles and old_engine_id:
                replaced_engine_ids.append(old_engine_id)
        for old_engine_id in replaced_engine_ids:
            self._retire_toolbox_registration(old_engine_id)

        toolboxes[tid] = {
            "toolbox_id": tid,
            "requests": [req.to_runtime_dict() for req in auto_requests],
            "manual_requests": [req.to_runtime_dict() for req in manual_requests],
            "profiles": new_profiles,
            "runtime": runtime,
            **(
                {
                    "intrinsics": {
                        "names": intrinsic_names,
                        "sandbox_profile": (intrinsic_profile or SandboxProfileSpec(profile_id="default")).to_dict(),
                        "with_intrinsic_guides": with_intrinsic_guides,
                    }
                }
                if intrinsic_names
                else {}
            ),
        }
        state["toolboxes"] = toolboxes
        self._write_toolboxes(state)
        removed_environment_keys = self._cleanup_unused_toolbox_environments(state)
        return {
            "status": "ok",
            "toolbox_id": tid,
            "request_count": len(manual_requests),
            "profiles": new_profiles,
            "spawned_engine_ids": spawned_engine_ids,
            "ready_engine_ids": list(ready_rollout.keys()),
            "rollout": ready_rollout,
            "replaced_engine_ids": replaced_engine_ids,
            "removed_environment_keys": removed_environment_keys,
        }

    def toolbox_unregister_manual(
        self,
        *,
        toolbox_id: str,
        tool_keys: List[str],
        python_executable: Optional[str] = None,
        worker_profile_class: str = "generic",
    ) -> Dict[str, Any]:
        tid = str(toolbox_id or "").strip()
        if not tid:
            raise ValueError("toolbox_id is required")
        return self._run_locked_toolbox_call(
            tid,
            self._toolbox_unregister_manual_impl,
            toolbox_id=tid,
            tool_keys=list(tool_keys or []),
            python_executable=python_executable,
            worker_profile_class=worker_profile_class,
        )

    def _toolbox_unregister_manual_impl(
        self,
        *,
        toolbox_id: str,
        tool_keys: List[str],
        python_executable: Optional[str] = None,
        worker_profile_class: str = "generic",
    ) -> Dict[str, Any]:
        from ..toolbox_harness import (
            SandboxProfileSpec,
            ToolboxAutoAssignmentRequest,
            ToolboxBundleStager,
            ToolboxManualAssignmentRequest,
            ToolboxSandboxOrchestrator,
        )

        tid = str(toolbox_id or "").strip()
        if not tid:
            raise ValueError("toolbox_id is required")
        keys = [str(item or "").strip() for item in list(tool_keys or []) if str(item or "").strip()]
        if not keys:
            raise ValueError("tool_keys are required")
        manual_requests, state, toolboxes = self._merge_toolbox_manual_requests(
            toolbox_id=tid,
            remove_keys=keys,
        )
        toolbox_row_existing = dict(toolboxes.get(tid) or {})
        auto_requests = [
            ToolboxAutoAssignmentRequest.from_runtime_dict(dict(item or {}))
            for item in list(toolbox_row_existing.get("requests") or [])
        ]
        intrinsics_row = dict(toolbox_row_existing.get("intrinsics") or {})
        intrinsic_names = self._normalize_intrinsic_tool_names(
            [str(item or "").strip() for item in list(intrinsics_row.get("names") or []) if str(item or "").strip()],
            include_guides=bool(intrinsics_row.get("with_intrinsic_guides", False)),
        )
        intrinsic_profile = SandboxProfileSpec.from_dict(dict(intrinsics_row.get("sandbox_profile") or {})) if intrinsic_names else None
        with_intrinsic_guides = bool(intrinsics_row.get("with_intrinsic_guides", False))
        existing_profiles = dict(dict(toolboxes.get(tid) or {}).get("profiles") or {})
        runtime = self._toolbox_runtime_defaults(
            toolbox_row_existing,
            python_executable=python_executable,
            worker_profile_class=worker_profile_class,
        )
        old_regs_by_profile: Dict[str, str] = {}
        for reg in self._toolbox_executor_registrations(tid):
            old_regs_by_profile[self._registration_sandbox_profile_id(reg)] = str(reg.get("engine_id") or "").strip()

        replaced_engine_ids: List[str] = []
        spawned_engine_ids: List[str] = []
        new_profiles: Dict[str, Dict[str, Any]] = {}
        if auto_requests or manual_requests or intrinsic_names:
            orchestrator = ToolboxSandboxOrchestrator(
                service=self,
                stager=ToolboxBundleStager(self.hosting_root),
                python_executable=runtime.get("python_executable"),
            )
            assignments = orchestrator.spawn_assignments(
                toolbox_id=tid,
                requests=auto_requests,
                manual_requests=manual_requests,
                intrinsic_tool_names=intrinsic_names,
                intrinsic_profile=intrinsic_profile,
                with_intrinsic_guides=with_intrinsic_guides,
                worker_profile_class=str(runtime.get("worker_profile_class") or "generic"),
            )
            try:
                ready_rollout = self._ensure_toolbox_assignments_ready(assignments, timeout_seconds=8.0)
            except Exception:
                for item in assignments:
                    reg = dict(item.registration or {})
                    engine_id = str(reg.get("engine_id") or "").strip()
                    if engine_id:
                        self._retire_toolbox_registration(engine_id)
                self._cleanup_unused_toolbox_environments(state)
                raise
            for item in assignments:
                profile_id = item.sandbox_profile.normalized_profile_id()
                reg = dict(item.registration or {})
                engine_id = str(reg.get("engine_id") or "").strip()
                if engine_id:
                    spawned_engine_ids.append(engine_id)
                old_engine_id = str(old_regs_by_profile.get(profile_id) or "").strip()
                bundle_revision = str(dict(reg.get("bundle") or {}).get("bundle_revision") or "")
                if old_engine_id and old_engine_id != engine_id:
                    replaced_engine_ids.append(old_engine_id)
                profile_auto_requests = [
                    req.to_runtime_dict()
                    for req in auto_requests
                    if req.sandbox_profile.normalized_profile_id() == profile_id
                ]
                profile_manual_requests = [
                    req.to_runtime_dict()
                    for req in manual_requests
                    if req.sandbox_profile.normalized_profile_id() == profile_id
                ]
                new_profiles[profile_id] = {
                    "sandbox_profile": item.sandbox_profile.to_dict(),
                    "requests": profile_auto_requests,
                    "manual_requests": profile_manual_requests,
                    "engine_id": engine_id,
                    "bundle_revision": bundle_revision,
                    "environment": dict(reg.get("environment") or {}),
                    "rollout": dict(ready_rollout.get(engine_id) or {}),
                    "rollout_history": self._append_toolbox_rollout_history(
                        dict(existing_profiles.get(profile_id) or {}),
                        rollout=dict(ready_rollout.get(engine_id) or {}),
                        action="unregister_manual",
                        engine_id=engine_id,
                        bundle_revision=bundle_revision,
                        replaced_engine_id=old_engine_id,
                    ),
                }
            for profile_id, old_engine_id in old_regs_by_profile.items():
                if profile_id not in new_profiles and old_engine_id:
                    replaced_engine_ids.append(old_engine_id)
            for old_engine_id in replaced_engine_ids:
                self._retire_toolbox_registration(old_engine_id)
            toolboxes[tid] = {
                "toolbox_id": tid,
                "requests": [req.to_runtime_dict() for req in auto_requests],
                "manual_requests": [req.to_runtime_dict() for req in manual_requests],
                "profiles": new_profiles,
                "runtime": runtime,
                **(
                    {
                        "intrinsics": {
                            "names": intrinsic_names,
                            "sandbox_profile": (intrinsic_profile or SandboxProfileSpec(profile_id="default")).to_dict(),
                            "with_intrinsic_guides": with_intrinsic_guides,
                        }
                    }
                    if intrinsic_names
                    else {}
                ),
            }
        else:
            ready_rollout = {}
            for old_engine_id in old_regs_by_profile.values():
                if old_engine_id:
                    replaced_engine_ids.append(old_engine_id)
            for old_engine_id in replaced_engine_ids:
                self._retire_toolbox_registration(old_engine_id)
            toolboxes.pop(tid, None)
        state["toolboxes"] = toolboxes
        self._write_toolboxes(state)
        removed_environment_keys = self._cleanup_unused_toolbox_environments(state)
        return {
            "status": "ok",
            "toolbox_id": tid,
            "remaining_request_count": len(manual_requests),
            "profiles": new_profiles,
            "spawned_engine_ids": spawned_engine_ids,
            "ready_engine_ids": list(ready_rollout.keys()),
            "rollout": ready_rollout,
            "replaced_engine_ids": replaced_engine_ids,
            "removed_tool_keys": keys,
            "toolbox_removed": not auto_requests and not manual_requests and not intrinsic_names,
            "removed_environment_keys": removed_environment_keys,
        }
