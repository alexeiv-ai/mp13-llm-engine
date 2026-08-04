from __future__ import annotations

import copy
import json

from app.context_cursor import ChatContext
from app.engine_session import EngineSession, InferenceParams
from mp13_engine.mp13_toolbox import Toolbox


def _context() -> tuple[EngineSession, ChatContext]:
    session = EngineSession()
    chat = session.add_conversation(inference_defaults=InferenceParams(), initial_params={})
    return session, ChatContext(session, chat_session=chat, toolbox=Toolbox())


def _active_anchor(context: ChatContext, name: str = "anchor-1", *, scope=None):
    cursor = context.active_cursor if scope is None else context.active_cursor_for_scope(scope)
    cursor.add_user(f"prompt for {name}")
    anchor = context.start_try_out_anchor(
        name,
        cursor.head,
        kind="auto_tool",
        retry_limit=3,
        origin_cursor=cursor,
        scope=scope,
    )
    cursor.add_try_out(anchor=anchor)
    return anchor


def test_active_anchor_round_trips_reconciles_and_closes_exactly_once() -> None:
    session, context = _context()
    _active_anchor(context)
    restored = EngineSession.from_dict(session.to_dict_prop)
    restored_context = ChatContext(restored, chat_session=restored.conversations[0], toolbox=Toolbox())

    before_cursor_count = len(restored_context.cursors_snapshot())
    assert restored_context.list_unresolved_try_out_anchors()[0]["reconciliation"]["status"] == "not_reconciled"
    assert restored_context.reconcile_try_out_anchors()[0]["status"] == "reconciled"
    assert len(restored_context.cursors_snapshot()) == before_cursor_count
    anchor = restored_context.get_try_out_anchor("anchor-1")
    assert anchor is not None and anchor.retries_remaining == 3

    restored_context.close_try_out_anchor("anchor-1", dist_mode="none")
    assert restored_context.close_try_out_anchor("anchor-1", dist_mode="none") is None
    assert restored_context.list_unresolved_try_out_anchors() == []


def test_repeat_reconcile_is_idempotent_and_does_not_duplicate_runtime_anchor() -> None:
    session, context = _context()
    _active_anchor(context)
    restored = EngineSession.from_dict(session.to_dict_prop)
    restored_context = ChatContext(restored, chat_session=restored.conversations[0], toolbox=Toolbox())

    first = restored_context.reconcile_try_out_anchors()
    second = restored_context.reconcile_try_out_anchors()
    assert first[0]["status"] == second[0]["status"] == "reconciled"
    assert len(restored_context.try_out_anchors_snapshot()) == 1


def test_multiple_scopes_reconcile_without_cross_binding() -> None:
    session, context = _context()
    first = _active_anchor(context, "anchor-default")
    scope = context.create_scope(label="secondary")
    secondary_cursor = context.active_cursor.clone()
    context.adopt_cursor(secondary_cursor, alias="secondary-origin", make_active=True, scope=scope)
    secondary_cursor.add_user("secondary prompt")
    second = context.start_try_out_anchor(
        "anchor-secondary",
        secondary_cursor.head,
        kind="auto_continue",
        retry_limit=4,
        origin_cursor=secondary_cursor,
        scope=scope,
    )
    secondary_cursor.add_try_out(anchor=second)

    restored = EngineSession.from_dict(session.to_dict_prop)
    restored_context = ChatContext(restored, chat_session=restored.conversations[0], toolbox=Toolbox())
    results = restored_context.reconcile_try_out_anchors()
    assert {item["status"] for item in results} == {"reconciled"}
    recovered_first = restored_context.get_try_out_anchor(first.anchor_name, allow_foreign_scope=True)
    recovered_second = restored_context.get_try_out_anchor(second.anchor_name, allow_foreign_scope=True)
    assert recovered_first is not None and recovered_second is not None
    assert recovered_first.owner_scope is not recovered_second.owner_scope
    assert recovered_first.owner_scope.scope_id == "default"
    assert recovered_second.owner_scope.scope_id == scope.scope_id


def test_missing_and_ambiguous_ids_are_explicitly_interrupted() -> None:
    session, context = _context()
    _active_anchor(context)
    source = session.to_dict_prop
    missing_payload = copy.deepcopy(source)
    missing_payload["chat_sessions"][0]["try_out_anchor_descriptors"][0]["anchor_turn_id"] = "missing-turn"
    restored = EngineSession.from_dict(missing_payload)
    restored_context = ChatContext(restored, chat_session=restored.conversations[0], toolbox=Toolbox())
    result = restored_context.reconcile_try_out_anchors()[0]
    assert result == {"anchor_name": "anchor-1", "status": "interrupted", "reason": "anchor_turn_missing"}
    assert restored_context.get_try_out_anchor("anchor-1") is None

    ambiguous_payload = copy.deepcopy(source)
    anchor_id = ambiguous_payload["chat_sessions"][0]["try_out_anchor_descriptors"][0]["anchor_turn_id"]
    duplicate_target = next(
        row for row in ambiguous_payload["nodes"].values() if row.get("gen_id") and row.get("gen_id") != anchor_id
    )
    duplicate_target["gen_id"] = anchor_id
    ambiguous = EngineSession.from_dict(ambiguous_payload)
    ambiguous_context = ChatContext(ambiguous, chat_session=ambiguous.conversations[0], toolbox=Toolbox())
    ambiguous_result = ambiguous_context.reconcile_try_out_anchors()[0]
    assert ambiguous_result["status"] == "interrupted"
    assert ambiguous_result["reason"] == "anchor_turn_ambiguous"


def test_closed_anchors_are_not_reopened_automatically_but_manual_resurrection_remains() -> None:
    session, context = _context()
    _active_anchor(context)
    context.close_try_out_anchor("anchor-1", dist_mode="none")
    restored = EngineSession.from_dict(session.to_dict_prop)
    restored_context = ChatContext(restored, chat_session=restored.conversations[0], toolbox=Toolbox())
    assert restored_context.reconcile_try_out_anchors() == []
    assert restored_context.get_try_out_anchor("anchor-1") is None

    resurrected = restored_context.resurrect_try_out_anchor("anchor-1")
    assert resurrected is not None
    assert resurrected.anchor_name == "anchor-1"


def test_descriptor_size_is_independent_of_message_bytes_and_tree_depth() -> None:
    session, context = _context()
    anchor = _active_anchor(context)
    baseline = len(json.dumps(context.chat_session.try_out_anchor_descriptors, sort_keys=True))
    cursor = context.resolve_try_out_cursor(anchor)
    assert cursor is not None
    cursor.add_user("x" * 250_000)
    for index in range(100):
        cursor.add_assistant(f"depth-{index}")
        cursor.add_user(f"next-{index}")
    after = len(json.dumps(context.chat_session.try_out_anchor_descriptors, sort_keys=True))
    assert after == baseline


def test_session_without_descriptors_does_not_scan_historical_markers_automatically() -> None:
    session, context = _context()
    _active_anchor(context)
    payload = copy.deepcopy(session.to_dict_prop)
    payload["chat_sessions"][0].pop("try_out_anchor_descriptors", None)
    restored = EngineSession.from_dict(payload)
    restored_context = ChatContext(restored, chat_session=restored.conversations[0], toolbox=Toolbox())
    assert restored_context.list_unresolved_try_out_anchors() == []
    assert restored_context.reconcile_try_out_anchors() == []
    assert restored_context.try_out_anchors_snapshot() == []
