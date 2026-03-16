# Phase 7 Advanced Hardening (Risk-Gated Design)

Date: 2026-03-15
Status: Draft design (planned, not enabled by default)
Scope: Post-Phase-8 optional hardening in `src/hosting`

## 1. Purpose

1. Preserve Phase 7 exactly as planned in `hosting_access_plan.md`:
   - key rotation automation + replay-resistance enhancements
   - optional hardware-backed key storage
   - advanced anomaly detection/lockout
2. Keep baseline functional flows unchanged for local-only and SSH-tunnel operations.
3. Require explicit risk/impact analysis before any Phase 7 control is implemented or enabled.

## 2. Non-Goals

1. No default-on hardening changes in this phase draft.
2. No breaking runtime behavior changes without an approved threat-reduction case.
3. No expansion of trust assumptions beyond current clean-slate auth/authz model.

## 3. Risk Gate Contract

Every Phase 7 candidate must ship with:
1. Threat statement:
   - specific attacker action reduced by the feature
   - affected deployment profiles (local_only / ssh_tunnel_only / truly_remote)
2. Impact statement:
   - operational cost
   - usability impact
   - failure/lockout blast radius
3. Rollback plan:
   - disable path
   - data/state recovery expectations
4. Test delta:
   - new tests required
   - manual runbook updates required

If any of the above is missing, the feature remains `planned` and disabled.

## 4. Candidate A: Key Rotation + Replay Resistance

Potential controls:
1. Admin-initiated key rotation workflow with overlap window.
2. Bounded replay window for challenge/session artifacts.
3. Server-issued nonce journaling for sensitive command classes.

Acceptance gate:
1. Must not break existing bootstrap flow in `hosting_config`.
2. Must define emergency admin recovery for rotated/expired credentials.
3. Must include deterministic audit entries for rotate/start/commit/abort.

## 5. Candidate B: Hardware-Backed Key Storage (Optional)

Potential controls:
1. Optional adapter-based key provider (TPM/HSM/OS key vault).
2. Fallback to current file-based keyring when hardware provider is unavailable.
3. Explicit capability probe in setup diagnostics.

Acceptance gate:
1. Must remain optional and off by default.
2. Must not block existing software-key path.
3. Must document platform-specific prerequisites and failure modes.

## 6. Candidate C: Anomaly Detection + Adaptive Lockout

Potential controls:
1. Rate/behavior thresholds on auth failures and ownership takeovers.
2. Time-boxed lockouts scoped by key/session/source context.
3. High-severity audit stream for suspicious sequences.

Acceptance gate:
1. Must have bounded false-positive impact for local dev workflows.
2. Must include explicit admin override/unlock path.
3. Must provide observability fields in `auth-audit-list` output.

## 7. Delivery Guardrails

1. No Phase 7 feature is required for Phase 8-complete baseline operation.
2. Phase 7 defaults remain disabled until a feature-specific gate review passes.
3. Documentation-first sequencing:
   - threat model delta
   - rollout + rollback
   - test plan + manual validation commands

## 8. Validation Planning (When Implementation Starts)

1. Unit:
   - rotation state machine and replay-window checks
   - anomaly threshold and lockout release logic
2. Integration:
   - daemon/channel/CLI/HTTP parity for new denial or audit paths
3. Manual:
   - setup + reconfigure in local/tunnel/remote profiles
   - admin recovery drills (lost key, lockout, rollback)

## 9. Current Decision

1. Keep Phase 7 as `Planned`.
2. Use this document as the required gate template before implementing any Phase 7 control.
