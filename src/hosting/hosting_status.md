# Hosting Server Implementation Progress

Last updated: 2026-08-08

## Purpose

This file is the progress ledger for server-side implementation of
`hosting_access_plan.md`. It reports executed work; it does not redefine the
contracts or duplicate the plan. Durable behavior belongs in the project
contracts and worker documentation. Dependent-project migration requirements
belong in `HOSTING_CLIENT_BREAKING_CHANGES.md`.

## Update rules

1. Define the active slice from unchecked plan item IDs before implementation.
2. Record the exact focused and regression test commands required to close the
   slice before changing code.
3. Do not move a slice to Completed or check its plan boxes until its code,
   tests, durable documentation, and migration handoff changes are complete.
4. Commit one coherent completed slice at a time. Include its plan item IDs in
   the commit subject or body.
5. Add the Completed entry, test results, commit subject, and checked plan item
   IDs in the same slice commit. A commit SHA is intentionally not recorded
   because a commit cannot contain its own final hash.
6. Keep failed or deferred work under Active or Blocked. Do not represent
   partial implementation as completed progress.
7. Do not rewrite completed entries except to correct a factual error; append a
   dated correction instead.

## Overall status

Status: In progress

Phases 0 through 3 are complete. The public contract/migration handoff, strict
dependency pipeline, hermetic environment builder, definition planner, and
durable pinned plan repository are implemented and audited. Phase 4 atomic
rollout and routing work is in progress.

## Active slice

None. The next slice will predeclare resolved candidate rollout and atomic
routing work (P4-02 onward) before changing files.

## Completed slices

| Date | Plan items checked | Delivered outcome | Tests passed | Commit subject |
| --- | --- | --- | --- | --- |
| 2026-08-08 | P0-01, P0-02 | Froze all current parent/dependent mutation, environment, payload, persisted-state, and actual hosted import removal/catalog inputs in the transient migration handoff. | Both predeclared parent/dependent `rg` inventories passed. AST assertions covered all 19 old dispatch commands, every deprecated hosted-ref method, parent intrinsic NumPy/SymPy/NumExpr imports, dependent Matplotlib import, and stale Requests declaration. Clean environment, daemon integration, persistence/concurrency, and runtime regression categories: not applicable (documentation-only inventory; no runtime change). | `docs: freeze hosting inventories (P0-01 P0-02)` |
| 2026-08-08 | P0-03, P0-06, P0-07, P0-09, P0-10, P0-11 | Froze the normative typed public definition/plan/approval/read/apply contract, strict limits and codes, actor rules, per-toolbox scope, retention, projections, and client algorithm; linked the dependent handoff. | `python -m pytest tests/test_hosted_toolbox_contract_docs.py -q` -> 5 passed; forbidden-vocabulary `rg` -> no matches. Clean environment, daemon integration, persistence/concurrency, and runtime regression categories: not applicable (documentation-only contract slice). | `docs: freeze toolbox public contract` (item IDs in commit body) |
| 2026-08-08 | P0-04 | Added canonical/domain-separated identity helpers and fixed vectors for definitions, resolved profiles, environments, manifests, template locks, and custom locks; documented dependent digest responsibilities. | `python -m pytest tests/test_hosted_toolbox_identity.py tests/test_hosted_toolbox_contract_docs.py -q` -> 10 passed, including two fresh hash-seeded processes; focused bundle staging regressions -> 3 passed, 135 deselected. Clean environment, daemon integration, and persistence/concurrency: not applicable until helpers are wired into runtime/state. | `feat: add canonical toolbox identities (P0-04)` |
| 2026-08-08 | P0-08 | Added definition-apply execution identity and strict persisted monotonic operation progress, including recovery, terminal diagnostic placement, and an irreversible publication cancellation boundary. | Contract/repository/toolbox-doc command -> 52 passed, including repository recreation/interruption and multi-process idempotency; operation service/workflow regressions -> 15 passed. Clean-environment build/import: not applicable. | `feat: extend hosted operation progress (P0-08)` |
| 2026-08-08 | P0-05 | Froze template/package/artifact trust, roles/control methods, signed immutable publication, offline preseeding, supported targets/timeouts, lifecycle/audit, and cache retention; removed dependent installation authority in the handoff. | Contract-doc command -> 6 passed; exact forbidden-history search -> no matches. Clean environment, daemon, persistence/concurrency, and runtime regressions: not applicable (policy-only). | `docs: freeze template deployment policy (P0-05)` |
| 2026-08-08 | P0-13, P0-14, P0-15 | Froze the complete initial catalog/config/sandbox/readiness contract, isolated cross-worker `core` resolution, and exclusive model-runtime lock/status/authorization boundary; added exact dependent selection/readiness/removal rules. | Contract-doc command -> 10 passed; exact forbidden-history search -> no matches; Phase 0 contract/identity/operation audit -> 62 passed. Clean environment, daemon integration, persistence/concurrency, and model execution regressions: not applicable (documentation-only freeze). | `docs: freeze initial environment boundaries (P0-13 P0-14 P0-15)` |
| 2026-08-08 | P0-12 | Completed the dependent adoption handoff with old/new code, durable recovery and teardown flows, exact version-1 archive/rollback procedure, and an inventory-to-removal matrix. | Document suite -> 12 passed; identity/operation audit -> 52 passed; exit audit -> 21 public operation names, 19 commands, and six dependent file groups covered; durable-contract forbidden-history search -> no matches. Runtime/state-command/client-repository categories: not applicable (documentation-only handoff). | `docs: complete toolbox migration handoff (P0-12)` |
| 2026-08-08 | P1-01, P1-02 | Added frozen strict immutable template/lock/provenance models and a deterministic reviewed import/distribution catalog, seeded only from inventoried roots. | Catalog -> 16 passed; identity/runtime-key regressions -> 20 passed; compile/diff checks passed. Clean environment and daemon integration: not applicable (pure models/catalog). | `feat: add toolbox template catalog (P1-01 P1-02)` |
| 2026-08-08 | P1-03 | Moved intrinsic discovery/dependency knowledge into import-safe metadata, derived environment roots/profile identity from it, removed profile branching, and pinned direct SymPy with a current lock. | Metadata -> 7 passed; complete toolbox sandbox -> 138 passed; Poetry lock, compile, and diff checks passed. Clean isolated-template execution remains P1-11/P1-12. | `refactor: isolate intrinsic dependency metadata (P1-03)` |
| 2026-08-08 | P1-04, P1-05, P1-06 | Added deterministic staged-source AST evidence, reviewed/explicit PEP 440 requirement resolution, and smallest exact template or minimal custom-delta selection. | Dependency pipeline -> 15 passed; catalog/identity regressions -> 21 passed; Poetry lock, compile, and diff checks passed. Clean environment, daemon, persistence/concurrency, and runtime execution: not applicable (pure planning). | `feat: add toolbox dependency planner (P1-04 P1-05 P1-06)` |
| 2026-08-08 | P1-07 | Added strict fail-closed target/template, package allow/deny, custom approval, HTTPS index, intrinsic-completeness, and payload-authority validation. | Policy -> 11 passed; dependency/catalog regressions -> 31 passed; compile/diff checks passed. Clean environment, daemon, persistence/concurrency, and runtime execution: not applicable (pure policy). | `feat: add toolbox dependency policy (P1-07)` |
| 2026-08-08 | P1-08 | Added an atomic immutable catalog with signed manifest/artifact references, lifecycle/active revisions, bounded consumer projections/audit, and end-to-end authorized service/daemon/channel/CLI controls. | Catalog control -> 11 passed including multi-process/restart/auth/transport; existing channel/CLI/auth regressions -> 95 passed; compile/diff checks passed. Environment materialization: not applicable until P1-09. | `feat: add toolbox catalog control (P1-08)` |
| 2026-08-08 | P1-09 | Added admin-only durable target-host prewarm orchestration, strict materializer/receipt boundaries, persisted bounded verification progress, and exact-receipt-gated consumer readiness. | Prewarm -> 6 passed; operation/catalog -> 58 passed; channel/CLI/auth -> 95 passed; contract docs -> 12 passed; compile/diff checks passed. An initially predeclared command named two nonexistent test files and failed before collection; it was corrected in the active ledger to the repository's actual CLI/auth suites before closeout. | `feat: add durable template prewarm (P1-09)` |
| 2026-08-08 | P1-10 | Closed the strict Phase 1 behavior matrix with explicit prewarm role exclusion and exact SSH CLI payload routing on top of direct dependency/catalog/offline tests. | Dependency analysis/policy -> 26 passed; catalog/prewarm -> 18 passed; channel/CLI/auth -> 95 passed; diff checks passed. | `test: close toolbox dependency matrix (P1-10)` |
| 2026-08-08 | P1-11 | Shipped exact independent `core`/`py-compute` locks, strict resource identities, compute-only policy, normal setup/prewarm and receipt-gated readiness, and smallest-template planning. | Shipped templates -> 6 passed; planner/policy/catalog -> 42 passed; catalog/prewarm/setup-state -> 21 passed; contract docs -> 12 passed; compile/diff checks passed. Physical installation remains Phase 2. | `feat: ship initial toolbox templates (P1-11)` |
| 2026-08-08 | P1-12 | Unified verified-template resolution/receipts across toolbox, node, snippet, and helper while deriving distinct consumer bindings and preserving runtime/API boundaries. | Shared resolver/isolated probes -> 9 passed; dependency/shipped-template -> 32 passed; workflow/helper/operation -> 37 passed; contract docs -> 12 passed; compile/diff checks passed. An initially predeclared workflow filename did not exist and failed before collection; the active ledger was corrected to the actual contract/node/helper suites before closeout. | `feat: share verified template resolution (P1-12)` |
| 2026-08-08 | P1-13, Phase 1 exit | Added exact bounded model-runtime status plus generic selection denials, including healthy-installed denial; proved deterministic restart-safe resolution without worker/ledger mutation. | Model boundary -> 12 passed; resolver/policy/catalog -> 32 passed; channel/CLI/auth/security -> 133 passed; docs/shipped templates -> 18 passed; Phase 1 resolver audit -> 10 passed; compile/diff checks passed. | `feat: enforce exclusive model runtime (P1-13)` |
| 2026-08-08 | Phase 2 groundwork (no items checked) | Added the strict host-derived resolved environment input and frozen `hosting.toolbox.environment.v2` cache identity, removed environment-description service lookup from worker orchestration, and isolated no-bootstrap toolbox interpreter selection while preserving workflow fallback. P2 remains open until the physical builder replaces the legacy inheriting path. | Hermetic contract -> 6 passed; workflow/runtime regressions -> 29 passed; focused toolbox environment regressions -> 3 passed, 135 deselected; compile/diff checks passed. | `refactor: establish hermetic toolbox environment contract` |
| 2026-08-08 | P2-01 through P2-10, Phase 2 exit | Added the target-host offline hermetic builder, exact wheel/lock/probe receipts, atomic publication/quarantine, thread/process deduplication, reference/grace GC, complete independent derived locks, receipt-gated readiness, catalog prewarm adaptation, and configured-orchestrator verified interpreter routing. | Hermetic builder -> 11 passed including real venvs, two fresh processes, service configuration, quarantine, ambient exclusion, complete base-plus-delta, and GC; environment/prewarm/shipped templates -> 19 passed; workflow/runtime -> 29 passed; focused toolbox environment -> 3 passed, 135 deselected; compile/diff checks passed. | `feat: build hermetic toolbox environments (P2-01..P2-10)` |
| 2026-08-08 | P3-01 through P3-05 | Added strict canonical version-2 definition/request/dependency models, host-derived resolved profiles, per-toolbox advertised-name and file conflict validation, post-resolution grouping, and resolved-profile bundle manifests. | Definition planner -> 11 passed; dependency/identity -> 20 passed; focused bundle/profile/duplicate regressions -> 18 passed, 120 deselected; docs/compile/diff checks passed. The initially predeclared dependency-pipeline filename did not exist and was corrected to the repository's dependency-analysis suite before execution. | `feat: add toolbox definition planner (P3-01..P3-05)` |
| 2026-08-08 | P3-06, P3-07, Phase 3 exit | Added exact active/proposed profile classification and a strict atomic expiring plan repository pinned to definition/expected/catalog/policy and resolved bundle identities. | Plan repository -> 10 passed including restart, expiry/no-refresh, pin identity, corruption, two-process writes, and no side effects; planner/identity -> 16 passed; operation repository -> 23 passed; docs/compile/diff checks passed. | `feat: persist toolbox definition plans (P3-06 P3-07)` |
| 2026-08-08 | P4-01 | Completed durable definition-apply recovery and cancellation dispatch with an atomic persisted publication boundary and bounded candidate-cleanup terminal diagnostics. | Definition apply/contract/repository -> 51 passed including the cancel/publication race; operation/workflow regressions -> 28 passed; compile/diff checks passed. The initially named `tests/test_hosting_workflow_contract.py` did not exist and failed before collection; closeout used the repository's actual workflow operation and Python contract suites. | `feat: add durable definition apply operations (P4-01)` |

P0-01/P0-02 exact verification commands:

```powershell
rg -n 'register_auto_callable|register_python_callable|register_manual_tool|register_intrinsics|unregister_|resolve_sandbox|environment_(description|resolve|apply|realize|sync|lock|verify|execute)|toolbox-(register|unregister|environment)' src tests
rg -n 'register_auto_callable|register_python_callable|register_manual_tool|register_intrinsics|unregister_|resolve_sandbox|environment_(description|resolve|apply|realize|sync|lock|verify|execute)|toolbox-(register|unregister|environment)' O:/repos/mp13-docs
```

The AST verification was executed as an inline Python assertion over
`src/hosting/toolbox/hosted_ref.py`, both dispatch implementations,
`src/mp13_engine/mp13_tools_builtin.py`,
`O:/repos/mp13-docs/src/tools/examples.py`, and the migration handoff. It
reported:

```text
commands=19; deprecated_methods_checked; parent_imports=['.mp13_config', '__future__', 'codecs', 'dataclasses', 'importlib', 'json', 'numexpr', 'numpy', 're', 'sympy', 'typing']; starter_imports=['__future__', 'base64', 'io', 'math', 'matplotlib', 'pathlib', 'tools']
```

## Blocked or deferred work

None.

Record blocked work with the affected unchecked plan item IDs, evidence, impact,
and the condition required to resume it.

## Next slice

Continue Phase 4 resolved candidate rollout, readiness validation, atomic
active routing, draining, and rollback work (P4-02 onward).
