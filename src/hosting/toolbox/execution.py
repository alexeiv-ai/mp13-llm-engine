"""Native toolbox execution harness."""
from __future__ import annotations

import asyncio
import inspect
import uuid
from typing import Any, Callable, Dict, List, Optional, Sequence

from mp13_engine.mp13_config import InferenceResponse, ParserProfile, ToolCall, ToolCallBlock
from mp13_engine.mp13_toolbox import Toolbox, ToolsView
from mp13_engine.mp13_tools_parser import UnifiedToolIO

from .bundle_models import ToolboxHarnessConfig
from .callbacks import _HostedToolCallbackRelay, _request_hosted_tool_approval
from .cancellation import _is_coarse_cancel_execution_error
from .tools_view import (
    _apply_tool_constraints_in_view,
    _approval_timeout_seconds,
    _approve_tool_in_view,
    _coerce_approval_decision,
    _extract_scope_constraints,
    _persist_approved_tool,
    _persist_scope_constraints,
    _resolve_scope_ref_from_callback_context,
    serialize_tools_view,
)


class ToolboxExecutionHarness:
    def __init__(
        self,
        *,
        config: Optional[ToolboxHarnessConfig] = None,
        native_toolbox: Optional[Toolbox] = None,
        control_channel: Optional[Any] = None,
    ) -> None:
        self.config = config or ToolboxHarnessConfig()
        self.native_toolbox = native_toolbox
        self.control_channel = control_channel
        self._rr_index = 0
        self._rr_lock = asyncio.Lock()

    @staticmethod
    def _normalize_parallel_execution(
        payload: Optional[Dict[str, Any]],
        *,
        sandbox_pool: bool,
        execution_model: str,
    ) -> Dict[str, Any]:
        row = dict(payload or {})
        hosted_pool = dict(row.get("hosted_pool") or row.get("toolbox_pool") or {})
        metrics = dict(hosted_pool.get("metrics") or {})

        def _int_value(*keys: str) -> int:
            for key in keys:
                try:
                    value = int(row.get(key) if key in row else metrics.get(key) or 0)
                except Exception:
                    value = 0
                if value:
                    return max(0, value)
            return 0

        def _float_value(*keys: str) -> float:
            for key in keys:
                try:
                    value = float(row.get(key) if key in row else metrics.get(key) or 0.0)
                except Exception:
                    value = 0.0
                if value:
                    return max(0.0, value)
            return 0.0

        effective_max = _int_value("effective_max_concurrency", "desired_capacity")
        queue_policy = str(
            row.get("queue_policy")
            or metrics.get("queue_policy")
            or ("unbounded" if execution_model == "native_async" else "bounded")
        ).strip().lower()
        if queue_policy not in {"bounded", "fail_fast", "unbounded"}:
            queue_policy = "bounded" if sandbox_pool else "unbounded"
        return {
            "supported": bool(row.get("supported", True)),
            "async_within_executor": bool(row.get("async_within_executor", True)),
            "sandbox_pool": bool(row.get("sandbox_pool", sandbox_pool)),
            "effective_max_concurrency": effective_max,
            "queue_policy": queue_policy,
            "queue_depth": _int_value("queue_depth"),
            "queue_timeout_seconds": _float_value("queue_timeout_seconds"),
            "active_calls": _int_value("active_calls"),
            "queued_calls": _int_value("queued_calls"),
            "worker_process_count": _int_value("worker_process_count", "worker_count"),
            "execution_model": str(row.get("execution_model") or execution_model).strip() or execution_model,
        }

    async def describe(self) -> Dict[str, Any]:
        mode = str(self.config.mode or "native").strip().lower()
        if mode == "native":
            if self.native_toolbox is None:
                raise RuntimeError("native_toolbox_not_configured")
            names = sorted(list(self.native_toolbox._registered_tool_names()))
            return {
                "mode": "native",
                "executor_kind": "native_toolbox",
                "all_registered_tool_names": names,
                "parallel_execution": self._normalize_parallel_execution(
                    {}, sandbox_pool=False, execution_model="native_async"
                ),
            }
        engine_id = await self._select_engine_id()
        toolbox_id = str(self.config.sandbox_toolbox_id or "").strip()
        if toolbox_id:
            result = await asyncio.to_thread(self.control_channel.toolbox_describe, toolbox_id=toolbox_id)
        else:
            result = await asyncio.to_thread(self.control_channel.toolbox_describe, engine_id=engine_id)
        out = dict(result or {})
        out.setdefault("mode", "sandbox")
        out["parallel_execution"] = self._normalize_parallel_execution(
            dict(out.get("parallel_execution") or {}),
            sandbox_pool=len(self.config.sandbox_engine_ids) > 1,
            execution_model="threaded_worker",
        )
        return out

    async def _resolve_execution_limit(
        self,
        requested_max_concurrency: Optional[int],
        call_count: int,
    ) -> int:
        requested = requested_max_concurrency
        if requested is None:
            requested = self.config.max_concurrency
        try:
            requested_value = int(requested or 0)
        except Exception:
            requested_value = 0
        if requested_value <= 0:
            try:
                described = await self.describe()
                requested_value = int(dict(described.get("parallel_execution") or {}).get("effective_max_concurrency") or 0)
            except Exception:
                requested_value = 0
        if requested_value <= 0:
            return max(1, int(call_count or 1))
        return max(1, min(requested_value, max(1, int(call_count or 1))))

    @staticmethod
    def _supports_keyword(callable_obj: Any, name: str) -> bool:
        try:
            signature = inspect.signature(callable_obj)
        except (TypeError, ValueError):
            return True
        return name in signature.parameters or any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )

    @staticmethod
    def _transport_error_envelope(call: ToolCall, *, request_id: str, exc: BaseException) -> Dict[str, Any]:
        reason = str(exc).strip() or "toolbox_transport_error"
        return {
            "status": "error",
            "outcome": "error",
            "reason": "transport_error",
            "error": reason,
            "error_type": type(exc).__name__,
            "request_id": request_id,
            "tool_call_id": str(call.id or "").strip() or None,
            "tool_name": str(call.name or "").strip(),
        }

    async def _execute_one_settled(
        self,
        call: ToolCall,
        *,
        semaphore: Optional[asyncio.Semaphore] = None,
        **kwargs: Any,
    ) -> ToolCall:
        try:
            if semaphore is None:
                return await self._execute_one(call, **kwargs)
            async with semaphore:
                return await self._execute_one(call, **kwargs)
        except Exception as exc:
            envelope = getattr(call, "execution_envelope", None)
            request_id = envelope.get("request_id") if isinstance(envelope, dict) else ""
            request_id = str(request_id or f"exec_{uuid.uuid4().hex}")
            call.error = call.error or f"Execution failed: {type(exc).__name__} - {exc}"
            call.execution_envelope = call.execution_envelope or self._transport_error_envelope(
                call, request_id=request_id, exc=exc
            )
            return call

    async def execute_calls(
        self,
        tool_calls: Sequence[ToolCall | Dict[str, Any]],
        *,
        parallel: bool = True,
        timeout_seconds: float = 30.0,
        native_execute_kwargs: Optional[Dict[str, Any]] = None,
        callback_processor: Optional[Callable[..., Any]] = None,
        callback_context: Any = None,
        host_api_approval: Optional[Dict[str, Any]] = None,
        max_concurrency: Optional[int] = None,
    ) -> List[ToolCall]:
        calls = [item if isinstance(item, ToolCall) else ToolCall.from_dict(dict(item or {})) for item in list(tool_calls or [])]
        if not calls:
            return []
        semaphore = None
        if parallel:
            semaphore = asyncio.Semaphore(await self._resolve_execution_limit(max_concurrency, len(calls)))
        if not parallel:
            out: List[ToolCall] = []
            for call in calls:
                out.append(
                    await self._execute_one_settled(
                        call,
                        timeout_seconds=timeout_seconds,
                        native_execute_kwargs=dict(native_execute_kwargs or {}),
                        callback_processor=callback_processor,
                        callback_context=callback_context,
                        host_api_approval=dict(host_api_approval or {}) if isinstance(host_api_approval, dict) else None,
                    )
                )
            return out
        tasks = [
            self._execute_one_settled(
                call,
                semaphore=semaphore,
                timeout_seconds=timeout_seconds,
                native_execute_kwargs=dict(native_execute_kwargs or {}),
                callback_processor=callback_processor,
                callback_context=callback_context,
                host_api_approval=dict(host_api_approval or {}) if isinstance(host_api_approval, dict) else None,
                )
            for call in calls
        ]
        return list(await asyncio.gather(*tasks, return_exceptions=False))

    async def execute_request_tools(
        self,
        parser_profile: ParserProfile,
        final_response_items: List[InferenceResponse],
        action_handler: Callable[..., Any],
        serial_execution: bool = False,
        *,
        tools_view: Optional[ToolsView] = None,
        context: Optional[Any] = None,
        tool_retries_max: Optional[int] = None,
        tool_retries_left: Optional[int] = None,
        timeout_seconds: float = 30.0,
        callback_processor: Optional[Callable[..., Any]] = None,
        callback_context: Any = None,
        host_api_approval: Optional[Dict[str, Any]] = None,
        max_concurrency: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        mode = str(self.config.mode or "native").strip().lower()
        if mode == "native" and self.native_toolbox is not None:
            await self.native_toolbox.execute_request_tools(
                parser_profile=parser_profile,
                final_response_items=final_response_items,
                action_handler=action_handler,
                serial_execution=serial_execution,
                tools_view=tools_view,
                context=context,
                tool_retries_max=tool_retries_max,
                tool_retries_left=tool_retries_left,
                **kwargs,
            )
            return

        all_blocks_to_parse: List[ToolCallBlock] = []
        for response_item in list(final_response_items or []):
            if response_item.tool_blocks and len(response_item.tool_blocks) > 0:
                for block in response_item.tool_blocks:
                    if block.prompt_index is None:
                        block.prompt_index = response_item.prompt_index
                all_blocks_to_parse.extend(response_item.tool_blocks)

        if not all_blocks_to_parse:
            return

        parser = UnifiedToolIO(profile=parser_profile)
        parser.parse_collected_blocks(all_blocks_to_parse)

        parsed_kwargs: Dict[str, Any] = {
            **kwargs,
            "context": context,
            "final_response_items": final_response_items,
            "current_response_item": None,
            "parser": parser,
            "tool_call": None,
            "tool_call_block": None,
            "tools_view": tools_view,
            "tool_retries_max": tool_retries_max,
            "tool_retries_left": tool_retries_left,
            "serial_execution": serial_execution,
        }
        await action_handler(execute_stage="calls_parsed", **parsed_kwargs)
        approval_state: Dict[str, Any] = {
            "cache": {},
            "lock": asyncio.Lock(),
        }
        execution_semaphore: Optional[asyncio.Semaphore] = None
        if not serial_execution:
            execution_semaphore = asyncio.Semaphore(
                await self._resolve_execution_limit(max_concurrency, sum(len(block.calls or []) for block in all_blocks_to_parse))
            )

        async def _execute_and_handle(
            tool_call: ToolCall,
            *,
            response_item: InferenceResponse,
            block: ToolCallBlock,
        ) -> None:
            action_kwargs = {
                **kwargs,
                "context": context,
                "final_response_items": final_response_items,
                "current_response_item": response_item,
                "parser": parser,
                "tool_call_block": block,
                "tools_view": tools_view,
                "tool_retries_max": tool_retries_max,
                "tool_retries_left": tool_retries_left,
                "serial_execution": serial_execution,
            }
            try:
                await action_handler(execute_stage="call_starting", tool_call=tool_call, **action_kwargs)
                executed = await self._execute_one_settled(
                    tool_call,
                    semaphore=execution_semaphore,
                    timeout_seconds=float(timeout_seconds or 30.0),
                    native_execute_kwargs=dict(
                        kwargs,
                        context=context,
                        tools_view=tools_view,
                        tool_retries_max=tool_retries_max,
                        tool_retries_left=tool_retries_left,
                    ),
                    callback_processor=callback_processor,
                    callback_context=callback_context,
                    host_api_approval=dict(host_api_approval or {}) if isinstance(host_api_approval, dict) else None,
                    approval_state=approval_state,
                )
                tool_call.result = executed.result
                tool_call.error = executed.error
                tool_call.action = list(executed.action or [])
                tool_call.id = executed.id or tool_call.id
                tool_call.parse_errors = list(executed.parse_errors or tool_call.parse_errors or [])
                tool_call.raw = executed.raw or tool_call.raw
                tool_call.model_format = executed.model_format or tool_call.model_format
                tool_call.execution_envelope = dict(executed.execution_envelope or {}) or None
            except Exception as exc:
                if not tool_call.error:
                    tool_call.error = f"Execution failed: {type(exc).__name__} - {exc}"
            finally:
                await action_handler(execute_stage="call_finished", tool_call=tool_call, **action_kwargs)

        if serial_execution:
            for response_item in list(final_response_items or []):
                for block in list(response_item.tool_blocks or []):
                    if not block.calls and not block.is_incomplete:
                        block.error_block = "Tool calls list is empty."
                        if ToolCall.KeepRaw not in (block.action_block or []):
                            block.action_block = list(block.action_block or [])
                            block.action_block.append(ToolCall.KeepRaw)
                        continue
                    if ToolCall.Ignore in block.action_block:
                        continue
                    for tool_call in list(block.calls or []):
                        if ToolCall.Ignore in tool_call.action:
                            continue
                        await _execute_and_handle(tool_call, response_item=response_item, block=block)
        else:
            tasks: List[asyncio.Task[Any]] = []
            for response_item in list(final_response_items or []):
                for block in list(response_item.tool_blocks or []):
                    if not block.calls and not block.is_incomplete:
                        block.error_block = "Tool calls list is empty."
                        if ToolCall.KeepRaw not in (block.action_block or []):
                            block.action_block = list(block.action_block or [])
                            block.action_block.append(ToolCall.KeepRaw)
                        continue
                    if ToolCall.Ignore in block.action_block:
                        continue
                    for tool_call in list(block.calls or []):
                        if ToolCall.Ignore in tool_call.action:
                            continue
                        tasks.append(asyncio.create_task(_execute_and_handle(tool_call, response_item=response_item, block=block)))
            if tasks:
                await asyncio.gather(*tasks)

        await action_handler(execute_stage="all_finished", **parsed_kwargs)

    async def _execute_one(
        self,
        call: ToolCall,
        *,
        timeout_seconds: float,
        native_execute_kwargs: Dict[str, Any],
        callback_processor: Optional[Callable[..., Any]] = None,
        callback_context: Any = None,
        host_api_approval: Optional[Dict[str, Any]] = None,
        approval_state: Optional[Dict[str, Any]] = None,
    ) -> ToolCall:
        mode = str(self.config.mode or "native").strip().lower()
        execution_request_id = f"exec_{uuid.uuid4().hex}"
        if mode == "native":
            if self.native_toolbox is None:
                raise RuntimeError("native_toolbox_not_configured")
            result = await self.native_toolbox.execute(call, **dict(native_execute_kwargs or {}))
            if result is not None:
                call.result = result
            call.execution_envelope = {
                "status": "ok",
                "outcome": "ok",
                "request_id": execution_request_id,
                "tool_call_id": str(call.id or "").strip() or None,
                "tool_name": str(call.name or "").strip(),
                "execution_model": "native_async",
            }
            return call
        engine_id = await self._select_engine_id()
        toolbox_id = str(self.config.sandbox_toolbox_id or "").strip()
        requested_tools_view = native_execute_kwargs.get("tools_view")
        tools_view_payload = serialize_tools_view(requested_tools_view)
        gate_payload: Dict[str, Any] = {}
        if hasattr(self.control_channel, "toolbox_gate"):
            if toolbox_id:
                gate_payload = dict(
                    await asyncio.to_thread(
                        self.control_channel.toolbox_gate,
                        toolbox_id=toolbox_id,
                        tool_name=str(call.name or "").strip(),
                        tools_view=tools_view_payload,
                    )
                )
            else:
                gate_payload = dict(
                    await asyncio.to_thread(
                        self.control_channel.toolbox_gate,
                        engine_id=engine_id,
                        tool_name=str(call.name or "").strip(),
                        tools_view=tools_view_payload,
                    )
                )
        outcome = str(gate_payload.get("outcome") or "").strip().lower()
        if outcome and outcome != "allowed":
            if outcome == "gated_requires_confirmation":
                decision = ""
                approval_result: Any = None
                cache_key = str(call.name or "").strip()
                cache = dict(approval_state or {}).get("cache") if isinstance(approval_state, dict) else None
                cache_lock = dict(approval_state or {}).get("lock") if isinstance(approval_state, dict) else None
                cache_future: Optional[asyncio.Future[Any]] = None
                is_owner = False
                if cache_key and cache is not None and isinstance(cache_lock, asyncio.Lock):
                    async with cache_lock:
                        existing = cache.get(cache_key)
                        if isinstance(existing, asyncio.Future):
                            cache_future = existing
                        elif isinstance(existing, dict):
                            approval_result = dict(existing)
                            decision = _coerce_approval_decision(approval_result)
                        else:
                            cache_future = asyncio.get_running_loop().create_future()
                            cache[cache_key] = cache_future
                            is_owner = True
                    if not decision and cache_future is not None:
                        if is_owner:
                            timeout_value = _approval_timeout_seconds(callback_context)
                            try:
                                approval_result = await asyncio.wait_for(
                                    asyncio.to_thread(
                                        _request_hosted_tool_approval,
                                        processor=callback_processor,
                                        toolbox_id=toolbox_id,
                                        tool_name=cache_key,
                                        tool_call_id=str(call.id or "").strip(),
                                        tool_arguments=dict(call.arguments or {}),
                                        callback_context=callback_context,
                                        gate_payload=gate_payload,
                                        tools_view=requested_tools_view,
                                    ),
                                    timeout=timeout_value,
                                )
                                decision = _coerce_approval_decision(approval_result)
                            except Exception:
                                approval_result = {"decision": "deny"}
                                decision = "deny"
                            if not cache_future.done():
                                cache_future.set_result(approval_result)
                            async with cache_lock:
                                if decision in {"deny", "add_to_scope"}:
                                    cache[cache_key] = dict(approval_result or {"decision": decision})
                                elif cache.get(cache_key) is cache_future:
                                    cache.pop(cache_key, None)
                        else:
                            try:
                                approval_result = await cache_future
                                decision = _coerce_approval_decision(approval_result)
                            except Exception:
                                decision = "deny"
                if not decision:
                    timeout_value = _approval_timeout_seconds(callback_context)
                    try:
                        approval_result = await asyncio.wait_for(
                            asyncio.to_thread(
                                _request_hosted_tool_approval,
                                processor=callback_processor,
                                toolbox_id=toolbox_id,
                                tool_name=str(call.name or "").strip(),
                                tool_call_id=str(call.id or "").strip(),
                                tool_arguments=dict(call.arguments or {}),
                                callback_context=callback_context,
                                gate_payload=gate_payload,
                                tools_view=requested_tools_view,
                            ),
                            timeout=timeout_value,
                        )
                        decision = _coerce_approval_decision(approval_result)
                    except Exception:
                        approval_result = {"decision": "deny"}
                        decision = "deny"
                scope_constraints = _extract_scope_constraints(approval_result, str(call.name or "").strip())
                if decision == "allow_once":
                    updated_view = _approve_tool_in_view(requested_tools_view, str(call.name or "").strip(), mutate=False)
                    updated_view = _apply_tool_constraints_in_view(
                        updated_view,
                        str(call.name or "").strip(),
                        scope_constraints,
                        mutate=True,
                    )
                    tools_view_payload = serialize_tools_view(updated_view)
                elif decision == "add_to_scope":
                    _approve_tool_in_view(requested_tools_view, str(call.name or "").strip(), mutate=True)
                    _apply_tool_constraints_in_view(
                        requested_tools_view,
                        str(call.name or "").strip(),
                        scope_constraints,
                        mutate=True,
                    )
                    scope_ref = _resolve_scope_ref_from_callback_context(callback_context)
                    _persist_approved_tool(scope_ref, str(call.name or "").strip())
                    _persist_scope_constraints(scope_ref, str(call.name or "").strip(), scope_constraints)
                    tools_view_payload = serialize_tools_view(requested_tools_view)
                else:
                    reason = str(gate_payload.get("reason") or outcome).strip() or outcome
                    call.error = f"Execution gated: denied - {reason}:{str(call.name or '').strip()}"
                    call.execution_envelope = {
                        "status": "error",
                        "outcome": "error",
                        "reason": reason,
                        "request_id": execution_request_id,
                        "tool_call_id": str(call.id or "").strip() or None,
                        "tool_name": str(call.name or "").strip(),
                    }
                    return call
            else:
                reason = str(gate_payload.get("reason") or outcome).strip() or outcome
                call.error = f"Execution gated: {outcome} - {reason}:{str(call.name or '').strip()}"
                call.execution_envelope = {
                    "status": "error",
                    "outcome": "error",
                    "reason": reason,
                    "request_id": execution_request_id,
                    "tool_call_id": str(call.id or "").strip() or None,
                    "tool_name": str(call.name or "").strip(),
                }
                return call
        try:
            callback_binding = None
            if callable(callback_processor):
                if not hasattr(self, "_callback_relay"):
                    self._callback_relay = _HostedToolCallbackRelay()
                signature = None
                try:
                    described = await self.describe()
                    tool_meta = dict(described.get("tool_metadata") or {}).get(str(call.name or "").strip()) or {}
                    signature = dict(tool_meta.get("callback_signature") or {}) or None
                except Exception:
                    signature = None
                callback_binding = self._callback_relay.bind_session(
                    processor=callback_processor,
                    toolbox_id=toolbox_id,
                    tool_name=str(call.name or "").strip(),
                    tool_call_id=str(call.id or "").strip(),
                    tool_arguments=dict(call.arguments or {}),
                    callback_signature=signature,
                    user_context=callback_context,
                )
            tool_call_payload = call.to_dict()
            tool_call_payload.pop("execution_envelope", None)
            execute_kwargs = {
                "tool_call": tool_call_payload,
                "timeout_seconds": float(timeout_seconds or 30.0),
                "tools_view": tools_view_payload,
                "callback_binding": dict(callback_binding or {}) or None,
            }
            execute_method = self.control_channel.toolbox_execute
            if self._supports_keyword(execute_method, "execution_request_id"):
                execute_kwargs["execution_request_id"] = execution_request_id
            if isinstance(host_api_approval, dict):
                execute_kwargs["host_api_approval"] = dict(host_api_approval or {})
            if toolbox_id:
                rpc_out = await asyncio.to_thread(
                    execute_method,
                    toolbox_id=toolbox_id,
                    **execute_kwargs,
                )
            else:
                rpc_out = await asyncio.to_thread(
                    execute_method,
                    engine_id=engine_id,
                    **execute_kwargs,
                )
        except Exception as exc:
            if _is_coarse_cancel_execution_error(exc):
                call.error = f"Execution canceled: sandbox_recycled:{str(call.name or '').strip()}"
                call.execution_envelope = {
                    **self._transport_error_envelope(call, request_id=execution_request_id, exc=exc),
                    "outcome": "canceled",
                    "reason": "sandbox_recycled",
                }
                return call
            raise
        finally:
            if 'callback_binding' in locals() and callback_binding and hasattr(self, "_callback_relay"):
                self._callback_relay.release_session(str(callback_binding.get("session_token") or ""))
        payload = dict(rpc_out or {})
        payload.setdefault("request_id", execution_request_id)
        payload.setdefault("tool_call_id", str(call.id or "").strip() or None)
        payload.setdefault("tool_name", str(call.name or "").strip())
        call.execution_envelope = dict(payload)
        if str(payload.get("status") or "").strip().lower() == "error":
            reason = str(payload.get("reason") or payload.get("error") or payload.get("message") or "toolbox_execute_failed").strip()
            call.error = call.error or f"Execution failed: {reason}"
            return call
        tool_out = dict(payload.get("tool_call") or {})
        if not tool_out:
            call.error = call.error or "Execution failed: toolbox_response_missing_tool_call"
            return call
        result_call = ToolCall.from_dict(tool_out)
        result_call.execution_envelope = dict(payload)
        return result_call

    async def _select_engine_id(self) -> str:
        if str(self.config.sandbox_toolbox_id or "").strip():
            return ""
        if self.control_channel is None:
            raise RuntimeError("control_channel_not_configured")
        engine_ids = [str(item or "").strip() for item in list(self.config.sandbox_engine_ids or []) if str(item or "").strip()]
        if not engine_ids:
            raise RuntimeError("sandbox_engine_ids_required")
        if len(engine_ids) == 1 or str(self.config.sandbox_selection or "round_robin").strip().lower() != "round_robin":
            return engine_ids[0]
        async with self._rr_lock:
            engine_id = engine_ids[self._rr_index % len(engine_ids)]
            self._rr_index = (self._rr_index + 1) % max(1, len(engine_ids))
            return engine_id
