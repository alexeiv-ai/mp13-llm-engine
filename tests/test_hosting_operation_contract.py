from __future__ import annotations

import pytest

from hosting.operation_contract import (
    MAX_INLINE_RESULT_BYTES,
    HostedExecutionKind,
    HostedOperationLifecycle,
    HostedOperationRef,
    HostedOperationSelector,
    HostedOperationStatus,
    HostedResultOmission,
    HostedResultRef,
    hosted_execution_fingerprint,
)


FINGERPRINT_VECTOR = {
    "execution_kind": "toolbox",
    "selector": {"kind": "toolbox_id", "id": "demo"},
    "tool": {"arguments": {"a": 1, "z": "last"}, "name": "write"},
}
FINGERPRINT_EXPECTED = "sha256:b4f06456b899a57aa45902d2912a1176dcd7fcc7e6e2c312cd40e02150390268"


def _ref() -> HostedOperationRef:
    return HostedOperationRef(
        operation_id="op_abc123",
        request_id="request-1",
        execution_kind=HostedExecutionKind.TOOLBOX,
        selector=HostedOperationSelector(kind="toolbox_id", id="demo"),
        fingerprint=FINGERPRINT_EXPECTED,
        receipt_namespace="toolbox:demo",
    )


def test_fingerprint_vector_is_canonical_and_order_independent() -> None:
    assert hosted_execution_fingerprint(FINGERPRINT_VECTOR) == FINGERPRINT_EXPECTED
    reordered = {
        "tool": {"name": "write", "arguments": {"z": "last", "a": 1}},
        "selector": {"id": "demo", "kind": "toolbox_id"},
        "execution_kind": "toolbox",
    }
    assert hosted_execution_fingerprint(reordered) == FINGERPRINT_EXPECTED


def test_operation_ref_round_trips() -> None:
    ref = _ref()
    assert HostedOperationRef.from_dict(ref.to_dict()) == ref


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("contract", "hosting.operation_ref.v2", "operation_ref_contract_invalid"),
        ("operation_id", "not valid", "operation_id_invalid"),
        ("execution_kind", "other", "operation_execution_kind_invalid"),
        ("fingerprint", "0" * 64, "operation_fingerprint_must_be_canonical_sha256"),
        ("request_id", "x" * 257, "operation_request_id_too_large"),
    ],
)
def test_operation_ref_rejects_malformed_values(field: str, value: object, reason: str) -> None:
    payload = _ref().to_dict()
    payload[field] = value
    with pytest.raises(ValueError, match=reason):
        HostedOperationRef.from_dict(payload)


def test_operation_ref_rejects_unknown_and_contradictory_selector_fields() -> None:
    payload = _ref().to_dict()
    payload["owner"] = "untrusted"
    with pytest.raises(ValueError, match="operation_ref_unknown_fields:owner"):
        HostedOperationRef.from_dict(payload)
    payload = _ref().to_dict()
    payload["selector"] = {"kind": "environment_key", "id": "demo"}
    with pytest.raises(ValueError, match="operation_selector_kind_invalid"):
        HostedOperationRef.from_dict(payload)


def test_status_round_trips_with_each_terminal_representation() -> None:
    ref = _ref()
    values = (
        {"result": {"answer": 7}},
        {
            "result_ref": HostedResultRef(
                artifact_id="artifact_1",
                digest=hosted_execution_fingerprint({"answer": 7}),
                size_bytes=12,
                expires_at_ms=123456,
            )
        },
        {
            "result_omission": HostedResultOmission(
                digest=hosted_execution_fingerprint({"answer": 7}),
                size_bytes=12,
                reason="retention_not_permitted",
            )
        },
    )
    for terminal in values:
        status = HostedOperationStatus(
            operation=ref,
            lifecycle=HostedOperationLifecycle.TERMINAL_SUCCESS,
            request_id=ref.request_id,
            created_at_ms=1000,
            updated_at_ms=1200,
            dispatch_claimed_at_ms=1050,
            terminal_at_ms=1200,
            **terminal,
        )
        assert HostedOperationStatus.from_dict(status.to_dict()) == status


def test_status_rejects_conflicting_or_oversized_terminal_payload() -> None:
    ref = _ref()
    omission = HostedResultOmission(
        digest=hosted_execution_fingerprint({"answer": 7}),
        size_bytes=12,
        reason="retention_not_permitted",
    )
    with pytest.raises(ValueError, match="operation_terminal_payload_conflict"):
        HostedOperationStatus(
            operation=ref,
            lifecycle=HostedOperationLifecycle.TERMINAL_SUCCESS,
            request_id=ref.request_id,
            created_at_ms=1,
            updated_at_ms=2,
            result={"answer": 7},
            result_omission=omission,
        )
    with pytest.raises(ValueError, match="operation_inline_result_too_large"):
        HostedOperationStatus(
            operation=ref,
            lifecycle=HostedOperationLifecycle.TERMINAL_SUCCESS,
            request_id=ref.request_id,
            created_at_ms=1,
            updated_at_ms=2,
            result="x" * (MAX_INLINE_RESULT_BYTES + 1),
        )


def test_status_rejects_request_mismatch_and_unknown_fields() -> None:
    with pytest.raises(ValueError, match="operation_status_request_id_mismatch"):
        HostedOperationStatus(
            operation=_ref(),
            lifecycle=HostedOperationLifecycle.RUNNING,
            request_id="different",
            created_at_ms=1,
            updated_at_ms=2,
        )
    payload = HostedOperationStatus(
        operation=_ref(),
        lifecycle=HostedOperationLifecycle.RUNNING,
        request_id="request-1",
        created_at_ms=1,
        updated_at_ms=2,
    ).to_dict()
    payload["status"] = "legacy"
    with pytest.raises(ValueError, match="operation_status_unknown_fields:status"):
        HostedOperationStatus.from_dict(payload)


def test_reason_and_timestamp_bounds_are_enforced() -> None:
    with pytest.raises(ValueError, match="operation_reason_too_large"):
        HostedOperationStatus(
            operation=_ref(),
            lifecycle=HostedOperationLifecycle.TERMINAL_FAILURE,
            request_id="request-1",
            created_at_ms=1,
            updated_at_ms=2,
            reason="x" * 513,
        )
    with pytest.raises(ValueError, match="operation_updated_at_before_created_at"):
        HostedOperationStatus(
            operation=_ref(),
            lifecycle=HostedOperationLifecycle.RUNNING,
            request_id="request-1",
            created_at_ms=2,
            updated_at_ms=1,
        )
