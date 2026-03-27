# Phase 7 Advanced Hardening Evaluation (Scenario-Driven)

Date: 2026-03-22
Status: Planned and optional (not default-on)
Scope: Evaluation guidance only for post-Phase-8 hardening in `src/hosting`

## 1. Goal

1. Clarify where each Phase 7 control is supported, what value it adds, and what operational overhead it introduces.
2. Keep baseline operation unchanged for users that do not opt into Phase 7.
3. Keep risk claims explicit: Phase 7 primarily reduces exposure and improves detection/recovery; it does not claim full protection after local user-compromise.

## 2. Baseline Risk Boundary

1. Baseline and Phase 7 assume local host compromise under daemon user can bypass many local controls.
2. Under that condition, Phase 7 value is mostly:
   - reducing credential lifetime usefulness
   - reducing replay window and abuse scale
   - improving tamper/abuse visibility and recovery workflow
3. No Phase 7 feature is approved as "full elimination" for post-compromise local attacker behavior.

## 3. Supported Scenarios

1. `local_only`
   - single host usage with lowest overhead
2. `ssh_tunnel_only`
   - off-host operators via SSH tunnel while daemon remains loopback-bound
3. `truly_remote`
   - direct/proxied non-loopback remote serving with explicit ingress policy

## 4. Hardening Candidates and Value

### 4.1 Candidate A: Key Rotation + Replay Resistance

Threat effect:
1. Partially mitigates key theft and replay by reducing useful credential lifetime and replay window.
2. Does not eliminate abuse during active compromise window.

Value by scenario:
1. `local_only`: low to medium value; mostly for disciplined key hygiene.
2. `ssh_tunnel_only`: high value; recommended first Phase 7 control.
3. `truly_remote`: high value; baseline hardening candidate for remote operation.

Human overhead:
1. Medium.
2. Requires rotation schedule, overlap policy, and emergency recovery runbook.

### 4.2 Candidate B: Hardware-Backed Key Storage (Optional)

Threat effect:
1. Partially mitigates export/offline theft of private key material.
2. Does not eliminate misuse from live authorized session compromise.

Value by scenario:
1. `local_only`: low unless compliance-driven.
2. `ssh_tunnel_only`: medium for high-privilege keys.
3. `truly_remote`: medium to high when assurance/compliance requires stronger key custody.

Human overhead:
1. High.
2. Platform capability variance, recovery complexity, and fallback policy management.

### 4.3 Candidate C: Anomaly Detection + Adaptive Lockout

Threat effect:
1. Partially mitigates brute-force, replay bursts, and suspicious auth/takeover sequences.
2. Improves detection and response quality.
3. Does not eliminate attacker success for valid credentials or post-compromise local control.

Value by scenario:
1. `local_only`: low to medium (false-positive risk may outweigh benefit).
2. `ssh_tunnel_only`: medium to high when auth pressure exists.
3. `truly_remote`: high; strongly justified at higher attack volume.

Human overhead:
1. Medium to high.
2. Threshold tuning, alert response ownership, unlock procedures, and false-positive handling.

### 4.4 Candidate D: Delegated SSH Signing / Key Custody (Transition Aid)

Threat effect:
1. Partially mitigates client-side key sprawl.
2. Introduces concentration risk (delegated signer becomes high-value target).
3. Does not eliminate compromise risk; may increase blast radius if policy binding is weak.

Value by scenario:
1. `local_only`: situational transition aid.
2. `ssh_tunnel_only`: situational migration aid.
3. `truly_remote`: generally low unless paired with stronger custody and anomaly controls.

Human overhead:
1. High.
2. Strict policy binding, unlock TTL lifecycle, break-glass recovery, and rollback readiness.

## 5. Scenario-Based Recommendation Order

1. `local_only`
   - Default: no Phase 7.
   - Optional: A only if key hygiene needs improve; C/B/D usually not justified.
2. `ssh_tunnel_only`
   - Recommended order: A first, then C (conservative tuning), then B for privileged keys if needed.
   - D only for temporary migration pain, never as default baseline.
3. `truly_remote`
   - Recommended order: A + C as primary.
   - Add B where assurance/compliance requires it.
   - D only with strict gate review and explicit blast-radius acceptance.

## 6. Risk Gate Contract (Required Before Enablement)

Each candidate must include:
1. Threat statement
   - attacker behavior reduced
   - scenarios in-scope (`local_only`, `ssh_tunnel_only`, `truly_remote`)
2. Value statement
   - what is actually reduced (eliminated vs partially mitigated)
   - expected measurable outcome
3. Overhead statement
   - operator workload (runbook, rotation, tuning, support)
   - failure/lockout blast radius
4. Rollback and recovery
   - deterministic disable path
   - break-glass and state recovery expectations
5. Test delta
   - new unit/integration tests
   - manual validation updates (`hosting_config`, `--doctor`, audit checks)

If any section is missing, feature remains `planned` and disabled.

## 7. Current Decision

1. Phase 7 remains planned and optional.
2. No candidate is default-on.
3. Candidate A is the most broadly justified first step for non-local scenarios.
4. Candidate D remains transition-only and requires strict explicit opt-in.
