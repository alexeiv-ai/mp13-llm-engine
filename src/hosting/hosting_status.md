## Hosting Sandbox Status

Date: 2026-04-01
Scope: Windows-first sandbox groundwork, now re-scoped toward toolbox sandbox executors rather than trusted engine workers

### Scope Correction

The status has changed in one important way:

1. Trusted engine instances such as [engine_worker_ipc.py](/o:/repos/mp13-llm-engine/src/hosting/engine_worker_ipc.py) are no longer treated as the intended first production sandbox consumer.
2. The intended first consumer is now a sandboxed toolbox-compatible executor that stages user-defined tool bundles and runs them outside the trusted engine worker.
3. Existing broker/policy/launcher work remains useful as reusable sandbox infrastructure, but not as the final boundary for the trusted engine path.

### Implemented Groundwork

1. Added `hosting/sandbox/` package:
   - [__init__.py](/o:/repos/mp13-llm-engine/src/hosting/sandbox/__init__.py)
   - [policy.py](/o:/repos/mp13-llm-engine/src/hosting/sandbox/policy.py)
   - [launcher.py](/o:/repos/mp13-llm-engine/src/hosting/sandbox/launcher.py)
   - [windows.py](/o:/repos/mp13-llm-engine/src/hosting/sandbox/windows.py)
   - [broker_fs.py](/o:/repos/mp13-llm-engine/src/hosting/sandbox/broker_fs.py)
   - [broker_http.py](/o:/repos/mp13-llm-engine/src/hosting/sandbox/broker_http.py)
   - [worker_fs.py](/o:/repos/mp13-llm-engine/src/hosting/sandbox/worker_fs.py)
   - [worker_http.py](/o:/repos/mp13-llm-engine/src/hosting/sandbox/worker_http.py)
2. Introduced `WorkerSandboxPolicy` normalization and persistence in worker registrations.
3. Moved worker launch orchestration out of [engine_host_service.py](/o:/repos/mp13-llm-engine/src/hosting/engine_host_service.py) into sandbox launcher helpers.
4. Default spawn hygiene now uses `close_fds=True` when `inherit_parent_handles=false`.
5. Added Windows sandbox launcher path intended to use:
   - restricted token
   - Low Integrity Level
   - Job Object
6. Added host-side brokered filesystem enforcement:
   - root-scoped read/write/list/stat/mkdir helpers
   - traversal denial
   - registration-bound sandbox policy lookup
7. Added host-side brokered HTTP enforcement:
   - broker-only network mode check
   - host allowlist and URL-prefix allowlist enforcement
   - response size cap and header sanitization
8. Added daemon/CLI broker commands:
   - `sandbox-fs-list`
   - `sandbox-fs-read-text`
   - `sandbox-fs-write-text`
   - `sandbox-fs-mkdir`
   - `sandbox-fs-stat`
   - `sandbox-http-fetch`
9. Added first-class control-channel broker helpers on [engine_host_channel.py](/o:/repos/mp13-llm-engine/src/hosting/engine_host_channel.py):
   - `sandbox_fs_list`
   - `sandbox_fs_read_text`
   - `sandbox_fs_write_text`
   - `sandbox_fs_mkdir`
   - `sandbox_fs_stat`
   - `sandbox_http_fetch`
10. Added toolbox bundle staging and execution helpers:
   - [toolbox_harness.py](/o:/repos/mp13-llm-engine/src/hosting/toolbox_harness.py)
   - manifest-driven staged bundle model with `bundle_id`, `bundle_revision`, `manifest_hash`, and tool allowlists
   - host-side harness support for:
     - native toolbox execution without sandbox
     - async parallel execution within one executor
     - round-robin execution across a pool of sandbox executors
11. Added dedicated toolbox executor worker:
   - [toolbox_executor_ipc.py](/o:/repos/mp13-llm-engine/src/hosting/toolbox_executor_ipc.py)
   - existing IPC transport reused for:
     - `toolbox.describe`
     - `toolbox.execute`
12. Extended host registration metadata for toolbox executors:
   - `executor_kind`
   - `bundle`
   - `environment`
   - `tool_access`
   - `capabilities`
13. Added host-side toolbox authorization and RPC surfaces:
   - `EngineHostService.toolbox_describe(...)`
   - `EngineHostService.toolbox_execute(...)`
   - `EngineHostControlChannel.toolbox_describe(...)`
   - `EngineHostControlChannel.toolbox_execute(...)`
   - CLI/daemon commands:
     - `toolbox-describe`
     - `toolbox-execute`
14. Added generic toolbox-worker callback support in [toolbox_executor_ipc.py](/o:/repos/mp13-llm-engine/src/hosting/toolbox_executor_ipc.py):
    - `host.call`
    - execution context wrappers:
      - `context.host.call(...)`
      - `context.fs.*`
      - `context.http.fetch(...)`
15. Added structured toolbox-worker startup spec support:
    - [toolbox_harness.py](/o:/repos/mp13-llm-engine/src/hosting/toolbox_harness.py)
    - `ToolboxWorkerStartupSpec`
    - staged spec-file generation via `worker_env_with_startup_spec(...)`
    - toolbox worker startup can now resolve manifest/config from `MP13_TOOLBOX_WORKER_SPEC_PATH`
    - startup spec now carries:
      - worker id
      - manifest path
      - scratch root
      - engines state file
      - control state file
16. Toolbox executors still accept toolbox-specific host metadata env vars as compatibility fallback:
    - `MP13_TOOLBOX_EXECUTOR_ENGINE_ID`
    - `MP13_HOSTING_ENGINES_STATE_FILE`
    - `MP13_HOSTING_CONTROL_STATE_FILE`
17. Extended toolbox revision staging to carry intrinsic-tool state:
    - `with_intrinsics`
    - `with_intrinsic_guides`
    - `intrinsic_tool_names`
    - `active_intrinsic_tool_names`
    - `hidden_intrinsic_tool_names`
18. Updated toolbox executor materialization so intrinsic-only revisions are valid:
    - sandbox revisions can now load builtin tools from [mp13_tools_builtin.py](/o:/repos/mp13-llm-engine/src/mp13_engine/mp13_tools_builtin.py) without staged user Python files
    - `toolbox.describe` / `toolbox.execute` now treat active intrinsic tools as part of the staged executor inventory
19. Added manifest-driven auto callable discovery from staged modules:
    - staged revisions can now list module/callable pairs instead of supplying a manual tool definition
    - worker loads the staged module and uses `Toolbox.add_tool_callable(callable_name, search_scope=module.__dict__, ...)`
    - tool definition is derived inside the sandbox from signature and docstring using existing toolbox behavior
20. Added first-slice automatic sandbox assignment helpers in [toolbox_harness.py](/o:/repos/mp13-llm-engine/src/hosting/toolbox_harness.py):
    - `SandboxProfileSpec` can derive stable profile ids from required imports and sandbox policy
    - `ToolboxAutoAssignmentRequest` models `callable + permissions spec + imports`
    - `ToolboxSandboxOrchestrator` can group requests by profile, stage one revision per profile, and spawn routed executor registrations under one logical toolbox id
21. Added persistent host-side logical toolbox registration in [engine_host_service.py](/o:/repos/mp13-llm-engine/src/hosting/engine_host_service.py):
    - `toolbox_register_auto(...)`
    - persisted state file at `<hosting_root>/state/toolbox_sandboxes.json`
    - incremental merge of auto-callable requests into logical toolbox membership
    - per-profile replacement rollout with old registration cleanup
22. Added persistent host-side logical toolbox removal in [engine_host_service.py](/o:/repos/mp13-llm-engine/src/hosting/engine_host_service.py):
    - `toolbox_unregister_auto(...)`
    - per-profile rebuild when tools are removed
    - full logical toolbox teardown when the last tool is removed
    - retired bundle-root cleanup under `<hosting_root>/toolbox_bundles`
23. Exposed high-level logical toolbox lifecycle through control surfaces:
    - `EngineHostControlChannel.toolbox_register_auto(...)`
    - `EngineHostControlChannel.toolbox_unregister_auto(...)`
    - CLI/daemon commands:
      - `toolbox-register-auto`
      - `toolbox-unregister-auto`
24. Added first-slice host-managed toolbox environment identity in [toolbox_harness.py](/o:/repos/mp13-llm-engine/src/hosting/toolbox_harness.py):
    - `ToolboxEnvironmentSpec`
    - `ToolboxEnvironmentManager`
    - deterministic `venv_key` derivation from runtime hash, intrinsic dependency tier, required imports, and optional dependency lock hash
    - environment metadata roots under `<hosting_root>/toolbox_venvs/<venv_key>`
25. Updated toolbox sandbox orchestration to prefer the structured startup-spec path when spawning profile-specific executors:
    - startup spec now carries `venv_path`
    - spawned toolbox registrations now carry non-null environment metadata including `venv_key` and `venv_path`
26. Extended the environment slice into real host-created toolbox venv roots:
    - environment roots are now created with stdlib `venv`
    - toolbox workers are now spawned through the environment Python executable
    - compatible revisions reuse the same venv root by `venv_key`
    - unreferenced toolbox venv roots are garbage-collected when logical toolbox state no longer references them
27. Added first-slice readiness-gated toolbox rollout in [engine_host_service.py](/o:/repos/mp13-llm-engine/src/hosting/engine_host_service.py):
    - `toolbox_register_auto(...)` and `toolbox_unregister_auto(...)` now wait for newly spawned profile-specific executors to answer `toolbox.describe`
    - replaced registrations are retired only after the new executors become ready
    - failed warmup rolls back the new registrations instead of cutting over
28. Added a simpler toolbox-facing facade in [toolbox_harness.py](/o:/repos/mp13-llm-engine/src/hosting/toolbox_harness.py):
    - public type: `HostedToolBoxRef`
    - compatibility alias: `SandboxedToolboxFacade`
    - wraps `toolbox_register_auto(...)`, `toolbox_unregister_auto(...)`, `toolbox_describe(...)`, and `toolbox_execute(...)`
    - hides low-level auto-request shaping for common sandboxed callable registration flows
29. Added first-slice rollout observability in [engine_host_service.py](/o:/repos/mp13-llm-engine/src/hosting/engine_host_service.py):
    - register/unregister now return `rollout` metadata for newly readied executors
    - persisted logical toolbox profile state now records basic rollout metadata such as `ready_at` and `warmup_ms`
30. Extended the higher-level toolbox facade in [toolbox_harness.py](/o:/repos/mp13-llm-engine/src/hosting/toolbox_harness.py):
    - `HostedToolBoxRef.register_python_callable(...)`
    - callers can now register a real module-backed Python callable without manually supplying staged file/module metadata
31. Added bounded per-profile rollout history in [engine_host_service.py](/o:/repos/mp13-llm-engine/src/hosting/engine_host_service.py):
    - successful `register_auto` / `unregister_auto` cutovers now append history entries with action, engine ids, bundle revision, and warmup timing
32. Extended high-level sandbox toolbox lifecycle to builtin intrinsics:
    - `EngineHostService.toolbox_register_intrinsics(...)`
    - `EngineHostService.toolbox_unregister_intrinsics(...)`
    - control channel / daemon / CLI forwarding for the same operations
    - `HostedToolBoxRef.register_intrinsic_tools(...)`
    - `HostedToolBoxRef.unregister_intrinsic_tools(...)`
33. Extended high-level sandbox toolbox lifecycle to explicit manual tool definitions:
    - `EngineHostService.toolbox_register_manual(...)`
    - `EngineHostService.toolbox_unregister_manual(...)`
    - control channel / daemon / CLI forwarding for the same operations
    - `SandboxedToolboxFacade.register_manual_tool(...)`
    - `SandboxedToolboxFacade.unregister_manual_tool(...)`
34. Added first-slice named environment descriptions:
    - sandbox profiles now carry `environment_name`
    - host persists environment descriptions in toolbox sandbox state
    - host APIs now support:
      - list descriptions
      - upsert description
      - clone description
      - resolve missing packages for linked toolbox functions
    - environment realization now incorporates environment-description identity into `venv_key`
35. Added explicit environment-apply rollout for linked toolbox sandboxes:
    - `EngineHostService.toolbox_environment_apply(...)`
    - control channel / daemon / CLI forwarding for the same operation
    - `HostedToolBoxRef.apply_environment_description(...)`
    - persisted toolbox state now records runtime defaults so later environment-apply operations can rebuild linked profiles with the same runtime shape
    - applying an updated environment description now rebuilds affected toolbox profiles and refreshes their realized environment metadata
36. Fixed named-environment inheritance in realized sandbox environments:
    - effective package sets are now resolved through the base-env chain
    - package-gap resolution now reports lineage, direct configured packages, and effective inherited packages
    - realized environment identity now changes when a base environment changes a derived environment’s effective package set
    - applying a base environment now rebuilds toolboxes linked through derived environments too
37. Added explicit environment-realization metadata:
    - `EngineHostService.toolbox_environment_realize(...)`
    - control channel / daemon / CLI forwarding for the same operation
    - `HostedToolBoxRef.realize_environment(...)`
    - realization now writes provenance/planning metadata into each realized env root and mirrors it into persisted toolbox profile state
    - the current realization mode is intentionally `metadata_only`; it does not yet claim package installation
38. Added explicit environment-description sync from linked tool requirements:
    - `EngineHostService.toolbox_environment_sync_description(...)`
    - control channel / daemon / CLI forwarding for the same operation
    - `HostedToolBoxRef.sync_environment_description(...)`
    - sync can now update the current named env description or clone into a new one with missing packages added
    - sync can optionally chain apply/realize after the description update, while still remaining metadata-only with respect to actual installs
39. Added explicit environment install-plan emission:
    - `EngineHostService.toolbox_environment_prepare_install(...)`
    - control channel / daemon / CLI forwarding for the same operation
    - `HostedToolBoxRef.prepare_environment_install(...)`
    - plan emission now writes `requirements-planned.txt` plus install-plan metadata into the env root and persisted toolbox profile state
    - the emitted install plan includes a concrete pip command template, but does not execute it yet
40. Added explicit environment install execution:
    - `EngineHostService.toolbox_environment_execute_install(...)`
    - control channel / daemon / CLI forwarding for the same operation
    - `HostedToolBoxRef.execute_environment_install(...)`
    - execution is policy-gated by both explicit caller opt-in and effective environment `allow_online_install`
    - execution result metadata is now recorded in the env root and persisted toolbox profile state as `blocked` / `noop` / `ok` / `failed`
41. Added explicit install locking:
    - `EngineHostService.toolbox_environment_lock_install(...)`
    - control channel / daemon / CLI forwarding for the same operation
    - `HostedToolBoxRef.lock_environment_install(...)`
    - locking now writes `requirements-locked.txt` plus `install_lock` metadata into the env root and persisted toolbox profile state
    - install execution now prefers the locked requirements artifact and records `install_lock_hash` in the execution result
42. Added explicit install-lock verification:
    - `EngineHostService.toolbox_environment_verify_install_lock(...)`
    - control channel / daemon / CLI forwarding for the same operation
    - `HostedToolBoxRef.verify_environment_install_lock(...)`
    - verification now records `ok` / `missing` / `stale` status plus expected-vs-current lock hash metadata
    - install execution now blocks when the lock is stale
43. Added post-install receipt capture:
    - successful install execution now runs a follow-up `pip freeze`
    - the observed package list and hash are written into `environment.json` and persisted toolbox profile state as `install_receipt`
    - this is observational provenance only, not a resolver-backed lock guarantee
44. Added explicit install-receipt verification:
    - `EngineHostService.toolbox_environment_verify_install_receipt(...)`
    - control channel / daemon / CLI forwarding for the same operation
    - `HostedToolBoxRef.verify_environment_install_receipt(...)`
    - verification now compares the observed receipt against the locked package set and records `ok` / `missing` / `mismatch`
    - this strengthens post-install provenance, but still does not replace a resolver-backed lock model
45. Added explicit resolved-install locking:
    - `EngineHostService.toolbox_environment_resolve_install_lock(...)`
    - control channel / daemon / CLI forwarding for the same operation
    - `HostedToolBoxRef.resolve_environment_install_lock(...)`
    - host can now run `pip install --dry-run --report ...` to persist a stronger exact-package lock artifact as `resolved_install_lock`
    - install execution and receipt verification now prefer that resolved lock when present
46. Added first-slice app-facing hosted toolbox helpers:
    - lightweight helper module [hosted_toolbox_api.py](/o:/repos/mp13-llm-engine/src/app/hosted_toolbox_api.py)
    - `create_hosted_toolbox_ref(...)`
    - `register_hosted_tool_callable(...)`
    - [mp13chat.py](/o:/repos/mp13-llm-engine/src/app/mp13chat.py) now re-exports those helpers for wrapper convenience
47. Added selectable hosted toolbox execution wiring for the chat runtime:
    - `ToolboxExecutionHarness.execute_request_tools(...)` now supports the same parsed-block execution contract used by the in-process toolbox path
    - [hosted_toolbox_api.py](/o:/repos/mp13-llm-engine/src/app/hosted_toolbox_api.py) now also provides:
      - `create_hosted_toolbox_executor(...)`
      - `HostedToolExecutionRouter`
    - [hosted_tool_runtime.py](/o:/repos/mp13-llm-engine/src/app/hosted_tool_runtime.py) now provides `execute_tool_round_on_cursor(...)` for a lightweight real app-runtime slice over `ChatCursor` / `ChatContext`
    - [mp13chat.py](/o:/repos/mp13-llm-engine/src/app/mp13chat.py) now has:
      - `configure_hosted_toolbox_execution(...)`
      - `clear_hosted_toolbox_execution()`
48. Added first-slice toolbox call gating:
    - [mp13_toolbox.py](/o:/repos/mp13-llm-engine/src/mp13_engine/mp13_toolbox.py) now provides `Toolbox.gate_call(...)` for native/toolbox-facing gate decisions
    - [engine_host_service.py](/o:/repos/mp13-llm-engine/src/hosting/engine_host_service.py) now provides `toolbox_gate(...)`
    - control channel / daemon / CLI forwarding now expose `toolbox-gate`
    - [toolbox_harness.py](/o:/repos/mp13-llm-engine/src/hosting/toolbox_harness.py) now preflights hosted calls through gate checks before `toolbox_execute(...)`
    - hosted gate denials now surface as explicit `Execution gated: ...` results instead of generic execution failures
      - internal `_active_tool_executor()` routing
    - the two chat execution callsites now execute through the active executor, so chat can preserve local `ToolBoxRef` state while routing actual tool calls through hosted sandbox execution
    - the hosted tool-response branch in [mp13chat.py](/o:/repos/mp13-llm-engine/src/app/mp13chat.py) now delegates to the lightweight runtime helper instead of inlining all hosted-execution handling
    - focused app-facing tests now cover both the lightweight execution router and a real cursor/session-based hosted tool round without importing the full chat runtime

48. Broader verification was confirmed outside this environment:
    - `python -m pytest tests/test_hosting_toolbox_sandbox.py -q` -> `46 passed`
    - `python -m pytest tests/test_hosting_toolbox_sandbox.py tests/test_engine_host_channel.py tests/test_hosting_worker_sandbox.py -q` -> `69 passed`
    - `python -c "import app.mp13chat as m; print('ok', hasattr(m, '_handle_live_prompt'), hasattr(m, 'configure_hosted_toolbox_execution'))"` -> `ok True True`
    - this means the current `mp13chat` import path is viable in the user environment, even though it remained blocked in my local execution environment because of a Python 3.12 `transformers` / `torch.compile` import issue
49. Added explicit `mp13chat` hosted-demo startup plumbing:
    - new helper module [hosted_chat_demo.py](/o:/repos/mp13-llm-engine/src/app/hosted_chat_demo.py)
    - `mp13chat` now supports:
      - `--hosted-demo`
      - `--hosted-demo-toolbox-id`
      - `--hosted-demo-project-root`
      - `--hosted-demo-hosting-root`
    - demo mode now registers two hosted tools under different sandbox profiles:
      - `SimpleCalc`
      - `ProjectFilePeek`
      - `ExampleHttpPeek`
    - `SimpleCalc` uses a basic isolated hosted profile
    - `ProjectFilePeek` uses a different hosted profile with brokered read-only filesystem access to the selected project root
    - `ExampleHttpPeek` uses a third hosted profile with brokered HTTP and a URL-prefix allowlist (`https://example.com/`)
    - startup now prints suggested prompts for exercising the hosted demo mode
    - shutdown now tears down the hosted demo toolbox registrations/workers on chat exit
50. Added first prompt-layer hosted tool advertisement filtering in chat:
    - [hosted_toolbox_api.py](/o:/repos/mp13-llm-engine/src/app/hosted_toolbox_api.py) now caches the hosted-advertisable tool set from `toolbox_describe(...)` or explicit configuration
    - [mp13chat.py](/o:/repos/mp13-llm-engine/src/app/mp13chat.py) now narrows outgoing `tools` payloads to that hosted-executable set when hosted execution is active
    - hosted demo mode now passes its known tool set explicitly so the model sees only the demo-hosted tools instead of unrelated local tools such as `scriptable_calculator`
51. Added first hosted visibility diagnostics for chat:
    - `HostedToolExecutionRouter` now exposes a hosted toolbox summary including the hosted-visible tool set
    - [mp13chat.py](/o:/repos/mp13-llm-engine/src/app/mp13chat.py) now includes that hosted-visible tool set in tool-scope summaries and hosted-demo startup output
52. Added effective hosted-aware toolbox inspection helpers:
    - new helper module [hosted_tool_visibility.py](/o:/repos/mp13-llm-engine/src/app/hosted_tool_visibility.py)
    - [mp13chat.py](/o:/repos/mp13-llm-engine/src/app/mp13chat.py) `/t` enumeration now reports effective availability and execution path (`hosted`, `native`, `gated`, `hidden`)
    - [mp13chat.py](/o:/repos/mp13-llm-engine/src/app/mp13chat.py) `/t sc` now reports effective advertised/hidden/disabled state after hosted filtering, plus hosted-gated tools
53. Added explicit operational toolbox reconciliation sweep:
    - [engine_host_service.py](/o:/repos/mp13-llm-engine/src/hosting/engine_host_service.py) now provides `toolbox_gc()`
    - control channel / daemon / CLI now expose `toolbox-gc`
    - the sweep reconciles persisted logical toolbox state against live toolbox executor registrations
    - stale toolbox executor registrations are retired
    - unused staged bundle roots under `<hosting_root>/toolbox_bundles` are removed
    - unreferenced toolbox environments under `<hosting_root>/toolbox_venvs` are removed
54. Tightened rollout readiness semantics:
    - [engine_host_service.py](/o:/repos/mp13-llm-engine/src/hosting/engine_host_service.py) now requires executor-reported `tool_names` from `toolbox.describe(...)` to match the staged/registered allowlist before cutover
    - rollout metadata now also records `tool_inventory_ok`, `tool_count`, and `tool_names`
    - readiness failure now includes inventory mismatches, not just transport/process unavailability
55. Added explicit toolbox reference reporting:
    - [engine_host_service.py](/o:/repos/mp13-llm-engine/src/hosting/engine_host_service.py) now provides `toolbox_references()`
    - control channel / daemon / CLI now expose `toolbox-references`
    - the report distinguishes:
      - persisted logical toolbox profiles
      - live toolbox executor registrations
      - referenced vs stale engine registrations
      - referenced vs stale bundle roots
      - referenced vs stale toolbox environments
56. Added explicit toolbox consistency reporting:
    - [engine_host_service.py](/o:/repos/mp13-llm-engine/src/hosting/engine_host_service.py) now provides `toolbox_consistency()`
    - control channel / daemon / CLI now expose `toolbox-consistency`
    - the report checks referenced logical toolbox state for live mismatch conditions, including:
      - missing referenced live registrations
      - toolbox-id / sandbox-profile-id drift between persisted profile state and the referenced live registration
      - referenced live tool inventory drift vs the expected per-profile toolbox inventory
      - missing referenced environment roots or missing `environment.json` metadata
57. Added explicit toolbox repair/rebuild flow:
    - [engine_host_service.py](/o:/repos/mp13-llm-engine/src/hosting/engine_host_service.py) now provides `toolbox_repair(...)`
    - control channel / daemon / CLI now expose `toolbox-repair`
    - repair now rebuilds inconsistent toolbox executors from persisted logical toolbox state, rolls replacement executors through the existing readiness + inventory gate, updates persisted profile state, and retires replaced registrations
58. Added explicit toolbox reconcile flow:
    - [engine_host_service.py](/o:/repos/mp13-llm-engine/src/hosting/engine_host_service.py) now provides `toolbox_reconcile(...)`
    - control channel / daemon / CLI now expose `toolbox-reconcile`
    - reconcile now captures consistency before action, runs selective repair, runs stale-artifact cleanup, and returns a consistency snapshot after action
59. Tightened minimal rollout-policy completion around structured failure paths:
    - [engine_host_service.py](/o:/repos/mp13-llm-engine/src/hosting/engine_host_service.py) now raises `ToolboxRolloutError` with structured `error_code` and `error_details`
    - readiness/inventory failures now carry explicit failure metadata such as `failure_phase`, `engine_id`, expected vs actual tool inventory, timeout, and routed toolbox/profile context
    - daemon / CLI / Python control-channel paths now preserve that structure instead of collapsing rollout failures to plain strings
    - operational surfaces now also return lightweight `summary` blocks for references / consistency / repair / reconcile / gc outputs
60. Added lightweight server-oriented admin helper:
    - new [toolbox_admin.py](/o:/repos/mp13-llm-engine/src/hosting/toolbox_admin.py)
    - public type: `HostedToolboxAdmin`
    - helper methods:
      - `review_snapshot(...)`
      - `startup_reconcile(...)`
      - `periodic_consistency_check(...)`
      - `auto_repair_if_needed(...)`
    - the helper wraps the existing control/service contract rather than adding a new sandbox lifecycle model
61. Added explicit review-snapshot operator surface:
    - [engine_host_service.py](/o:/repos/mp13-llm-engine/src/hosting/engine_host_service.py) now provides `toolbox_review_snapshot(...)`
    - control channel / daemon / CLI now expose `toolbox-review-snapshot`
    - `HostedToolboxAdmin.review_snapshot(...)` now prefers that shared hosting-side contract when available
    - the snapshot combines references + consistency + compact summary + `recommended_action`
62. Added explicit serialization/deserialization for hosted sandbox toolbox proxies:
    - [toolbox_harness.py](/o:/repos/mp13-llm-engine/src/hosting/toolbox_harness.py) `HostedToolBoxRef` now provides `to_dict()` / `from_dict(...)`
    - current supported host descriptors cover:
      - control-channel-backed refs via serialized `control_settings`
      - service-backed refs via serialized state-file paths
    - this makes the remote thin-client workflow persistable without inventing a separate toolbox-proxy wire format
63. Tightened operator review/reference UX:
    - `toolbox-review-snapshot` now returns a compact per-toolbox summary with profile rows, issue names, and recommendation fields instead of repeating the full raw references tree
    - bundle-reference reporting and bundle GC now treat referenced revision subdirectories as keeping their parent profile bundle directories live, avoiding false `stale_bundle_roots` reports in hosted multi-revision layouts
64. Tightened reconcile UX:
    - `toolbox-reconcile` now returns a compact operator-oriented default payload with requested/target/repaired toolbox ids, removed artifact ids, summary counts, and a simple `outcome`
    - deep `before` / `repair` / `gc` / `after` internals are still available through `details=true`
65. Tightened repair UX:
    - `toolbox-repair` now also returns a compact operator-oriented default payload with requested/target/repaired/skipped toolbox ids, removed environment keys, summary counts, and a simple `outcome`
    - deep repaired/skipped internals are still available through `details=true`

### Current Interpretation Of That Work

What is implemented today should now be interpreted as sandbox infrastructure:

1. policy schema and support-label groundwork
2. Windows spawn-hardening and Low-IL starter enforcement
3. brokered filesystem and brokered HTTP policy enforcement on the host
4. helper surfaces for CLI/channel integration
5. an initial manifest-driven toolbox sandbox executor slice
6. host-side tool allowlist gating before `toolbox.execute` dispatch
7. parallel toolbox execution support through async dispatch, harness-managed executor pools, or both
8. preserved native toolbox execution mode when sandboxing is not required
9. initial generic host callback path for toolbox-worker brokered operations
10. initial structured startup-spec path for toolbox executors, with legacy env fallback still accepted
11. intrinsic-aware toolbox revisions, including intrinsic-only sandbox executors and intrinsic guide exposure
12. auto-discovered sandbox callables from staged modules using existing toolbox registration logic
13. first-slice host routing by logical `toolbox_id` across multiple sandbox-profile-specific executor registrations
14. first-slice automatic profile assignment and grouped spawn orchestration for auto-callable requests
15. first-slice persistent logical toolbox membership and incremental auto-registration updates
16. first-slice persistent logical toolbox removal and per-profile rebuild on unregister
17. first-slice host-managed environment identity and reuse across toolbox revisions with the same dependency profile
18. first-slice host-created toolbox venv roots, worker spawn through venv Python, and venv-root GC on unregister
19. first-slice readiness-gated cutover for toolbox registration updates and removals
20. first-slice higher-level facade for common sandboxed toolbox registration/execute flows
21. first-slice persisted rollout metadata for profile cutovers
22. first-slice direct Python callable registration through the facade
23. first-slice bounded rollout history for successful profile cutovers
24. first-slice intrinsic-tool registration/removal through the high-level sandbox facade and control surfaces
25. first-slice manual tool-definition registration/removal through the high-level sandbox facade and control surfaces
26. first-slice named environment-description persistence and package-gap resolution
27. first-slice explicit environment-apply rebuilds for linked toolbox sandboxes using persisted runtime defaults

What it is not yet:

1. a full bundle removal / garbage-collection lifecycle tied to active references
2. a finished immutable sandbox `.venv` management story
3. executor-side cancellation support (`toolbox.cancel`)
4. a trustworthy direct-network enforcement layer on Windows

### Decision On Access Gating

The current design direction is:

1. primary permission gating must happen on the host before dispatching sandbox execution
2. sandbox-side toolbox checks remain defense in depth only
3. `Toolbox.execute(...)` in [mp13_toolbox.py](/o:/repos/mp13-llm-engine/src/mp13_engine/mp13_toolbox.py) is not sufficient as the security boundary for sandboxed user tools
4. host registration metadata now carries the staged tool inventory used for pre-dispatch allowlist checks

Rationale:

1. host owns the trust boundary and caller/session context
2. host must decide whether a tool is staged, allowed, and callable before untrusted code starts
3. sandbox-side checks still matter, but they are secondary

### Decision On Add/Remove Tool Management

The current design direction is:

1. adding sandboxed tools is now represented as host-managed manifest staging, not ambient `search_scope` discovery
2. add and remove should be treated symmetrically as "new toolbox revision -> replacement worker rollout"
3. removing sandboxed tools should still deactivate host routing first, then unload/restart executor, then garbage-collect staged content when safe
4. dynamic tool registration APIs in [mp13_toolbox.py](/o:/repos/mp13-llm-engine/src/mp13_engine/mp13_toolbox.py) remain valid for trusted in-process workflows and the explicit native harness mode, but not as the authority path for sandbox management

Intrinsic tools need the same treatment:

1. builtins defined in [mp13_tools_builtin.py](/o:/repos/mp13-llm-engine/src/mp13_engine/mp13_tools_builtin.py) are part of the toolbox model, not an external exception
2. intrinsic enable/disable operations should therefore map to toolbox revision state under sandbox management
3. enabling an intrinsic guide tool should be represented explicitly in revision metadata
4. sandbox execution of builtin tools should still go through the same host authorization and replacement-worker rollout model as user-defined tools

### Decision On Worker Startup And Callback Contract

The current design direction is:

1. toolbox executors should use the generic hosting worker mechanism, not the trusted engine worker path
2. startup should move toward one structured startup spec rather than a growing list of ad hoc env vars
3. brokered toolbox callbacks should be solved as a generic host callback RPC design, with convenience wrappers on top
4. normal callers should still interact through a simple toolbox-facing API while hosting hides revision/restart churn

### Known Limitations That Must Remain Explicit

1. Windows Low IL is primarily a write-protection boundary; it does not provide strong same-account read isolation.
2. Brokered host/prefix allowlists are trustworthy only on the brokered HTTP path.
3. Direct worker networking on Windows remains `partial` or `unsupported` for first-cut policy claims.
4. “Selected route” network control is not a trustworthy first Windows promise unless stronger OS/network enforcement such as WFP/firewall/proxy mediation is added.
5. Sandbox `.venv` handling is not yet designed well enough to claim reproducible, immutable dependency isolation.
6. Live dependency installation during sandbox execution would weaken provenance and policy enforcement and should not be the first-cut model.
7. compatibility env vars still exist for toolbox-worker metadata, but the structured startup spec now carries manifest and hosting-state metadata on the preferred path

### Recommended `.venv` Direction

The current plan direction is:

1. host-built `.venv`
2. immutable after activation
3. keyed by bundle/dependency hash
4. writable scratch kept separate from `.venv`
5. builtin-tool dependency tiers must be included in the environment key
6. no claim yet that this is implemented

Current builtin dependency signal from [mp13_tools_builtin.py](/o:/repos/mp13-llm-engine/src/mp13_engine/mp13_tools_builtin.py):

1. `scriptable_calculator` depends on `numpy` and can optionally use `numexpr`
2. `symbolic_algebra` depends on `sympy`
3. guide tools are lighter logically, but their activation still belongs to the same toolbox revision model

So the status should now be read as:

1. enabling some intrinsic tools may require a different immutable `.venv`
2. builtin activation is not just a registry toggle; it can affect environment provenance
3. the exact intrinsic profile scheme is still a design item, not an implemented feature

### Clarified User-Facing Intent

Normal users should still be able to think in toolbox operations:

1. add tool
2. remove tool
3. execute tool
4. list tools

The hosting/runtime layer should hide:

1. revision generation
2. worker restart and switchover
3. staged path bookkeeping
4. future environment selection details

`ToolBoxRef` and `ToolsAccess` remain the logical access layer, but not the owner of sandbox process lifecycle.

### Test Evidence For Implemented Groundwork

Commands run:

1. `python -m pytest tests/test_hosting_worker_sandbox.py tests/test_hosting_daemon_pidfile.py tests/test_hosting_service_security.py tests/test_hosting_auth_roles.py -q`
   - result: `67 passed`
2. `python -m pytest tests/test_engine_host_channel.py tests/test_hosting_daemon_startup.py tests/test_hosting_daemon_pidfile.py -q`
   - result: `29 passed`
3. `python -m pytest tests/test_hosting_worker_sandbox.py tests/test_hosting_worker_sandbox_windows_live.py -q`
   - result: `11 passed, 1 skipped`
4. `python -m pytest tests/test_hosting_worker_sandbox.py tests/test_hosting_worker_sandbox_windows_live.py tests/test_hosting_daemon_pidfile.py tests/test_hosting_service_security.py tests/test_hosting_auth_roles.py tests/test_engine_host_channel.py tests/test_hosting_daemon_startup.py -q`
   - result: `87 passed, 1 skipped`
5. `python -m pytest tests/test_hosting_worker_sandbox.py -q`
   - result: `9 passed`
6. `python -m pytest tests/test_hosting_worker_sandbox.py tests/test_engine_host_channel.py tests/test_hosting_daemon_startup.py -q`
   - result: `21 passed`
7. `python -m pytest tests/test_engine_host_channel.py tests/test_hosting_worker_sandbox.py -q`
   - result: `22 passed`
8. `pytest tests/test_hosting_toolbox_sandbox.py -q -p no:tmpdir`
   - result: `5 passed`
9. `python -m pytest tests/test_hosting_toolbox_sandbox.py -q`
   - result: `11 passed`
10. `python -m pytest tests/test_hosting_toolbox_sandbox.py tests/test_hosting_worker_sandbox.py tests/test_engine_host_channel.py -q`
   - result: `33 passed`
11. `python -m pytest tests/test_hosting_toolbox_sandbox.py -q`
   - result: `17 passed`
12. `python -m pytest tests/test_hosting_toolbox_sandbox.py tests/test_engine_host_channel.py tests/test_hosting_worker_sandbox.py -q`
   - result: `39 passed`
13. `python -m pytest tests/test_hosting_toolbox_sandbox.py -q`
   - result: `19 passed`
14. `python -m pytest tests/test_hosting_toolbox_sandbox.py tests/test_engine_host_channel.py tests/test_hosting_worker_sandbox.py -q`
   - result: `41 passed`
15. `python -m pytest tests/test_hosting_toolbox_sandbox.py -q`
   - result: `22 passed`
16. `python -m pytest tests/test_hosting_toolbox_sandbox.py tests/test_engine_host_channel.py tests/test_hosting_worker_sandbox.py -q`
   - result: `44 passed`
17. `python -m pytest tests/test_hosting_toolbox_sandbox.py -q`
   - result: `23 passed`
18. `python -m pytest tests/test_hosting_toolbox_sandbox.py tests/test_engine_host_channel.py tests/test_hosting_worker_sandbox.py -q`
   - result: `45 passed`
19. `python -m pytest tests/test_hosting_toolbox_sandbox.py -q`
   - result: `25 passed`
20. `python -m pytest tests/test_hosting_toolbox_sandbox.py tests/test_engine_host_channel.py tests/test_hosting_worker_sandbox.py -q`
   - result: `47 passed`
21. `python -m pytest tests/test_engine_host_channel.py -q`
   - result: `14 passed`
22. `python -m pytest tests/test_hosting_toolbox_sandbox.py tests/test_engine_host_channel.py tests/test_hosting_worker_sandbox.py -q`
   - result: `48 passed`
23. `python -m pytest tests/test_hosting_toolbox_sandbox.py -q`
   - result: `27 passed`
24. `python -m pytest tests/test_engine_host_channel.py -q`
   - result: `14 passed`
25. `python -m pytest tests/test_hosting_toolbox_sandbox.py tests/test_engine_host_channel.py tests/test_hosting_worker_sandbox.py -q`
   - result: `50 passed`
26. `python -m pytest tests/test_hosting_toolbox_sandbox.py -q`
   - result: `28 passed`
27. `python -m pytest tests/test_hosting_toolbox_sandbox.py tests/test_engine_host_channel.py tests/test_hosting_worker_sandbox.py -q`
   - result: `51 passed`
28. `python -m pytest tests/test_hosting_toolbox_sandbox.py -q`
   - result: `30 passed`
29. `python -m pytest tests/test_engine_host_channel.py -q`
   - result: `14 passed`
30. `python -m pytest tests/test_hosting_toolbox_sandbox.py tests/test_engine_host_channel.py tests/test_hosting_worker_sandbox.py -q`
   - result: `53 passed`
31. `python -m pytest tests/test_hosting_toolbox_sandbox.py -q`
   - result: `31 passed`
32. `python -m pytest tests/test_engine_host_channel.py -q`
   - result: `14 passed`
33. `python -m pytest tests/test_hosting_toolbox_sandbox.py tests/test_engine_host_channel.py tests/test_hosting_worker_sandbox.py -q`
   - result: `54 passed`
34. `python -m pytest tests/test_hosting_toolbox_sandbox.py -q`
   - result: `33 passed`
35. `python -m pytest tests/test_hosting_toolbox_sandbox.py tests/test_engine_host_channel.py tests/test_hosting_worker_sandbox.py -q`
   - result: `56 passed`
36. `python -m pytest tests/test_hosting_toolbox_sandbox.py -q`
   - result: `34 passed`
37. `python -m pytest tests/test_hosting_toolbox_sandbox.py tests/test_engine_host_channel.py tests/test_hosting_worker_sandbox.py -q`
   - result: `57 passed`
38. `python -m pytest tests/test_hosting_toolbox_sandbox.py -q`
   - result: `35 passed`
39. `python -m pytest tests/test_hosting_toolbox_sandbox.py tests/test_engine_host_channel.py tests/test_hosting_worker_sandbox.py -q`
   - result: `58 passed`
40. `python -m pytest tests/test_hosting_toolbox_sandbox.py -q`
   - result: `36 passed`
41. `python -m pytest tests/test_hosting_toolbox_sandbox.py tests/test_engine_host_channel.py tests/test_hosting_worker_sandbox.py -q`
   - result: `59 passed`
42. `python -m pytest tests/test_engine_host_channel.py tests/test_hosting_toolbox_sandbox.py -q`
   - result: `50 passed`
43. `python -m pytest tests/test_hosting_toolbox_sandbox.py tests/test_engine_host_channel.py tests/test_hosting_worker_sandbox.py -q`
   - result: `59 passed`
44. `python -m pytest tests/test_hosting_toolbox_sandbox.py -q`
   - result: `38 passed`
45. `python -m pytest tests/test_hosting_toolbox_sandbox.py tests/test_engine_host_channel.py tests/test_hosting_worker_sandbox.py -q`
   - result: `61 passed`
46. `python -m pytest tests/test_engine_host_channel.py tests/test_hosting_toolbox_sandbox.py -q`
   - result: `53 passed`
47. `python -m pytest tests/test_hosting_toolbox_sandbox.py tests/test_engine_host_channel.py tests/test_hosting_worker_sandbox.py -q`
   - result: `62 passed`
48. `python -m pytest tests/test_engine_host_channel.py tests/test_hosting_toolbox_sandbox.py -q`
   - result: `55 passed`
49. `python -m pytest tests/test_hosting_toolbox_sandbox.py tests/test_engine_host_channel.py tests/test_hosting_worker_sandbox.py -q`
   - result: `64 passed`
50. `python -m pytest tests/test_engine_host_channel.py tests/test_hosting_toolbox_sandbox.py -q`
   - result: `56 passed`
51. `python -m pytest tests/test_hosting_toolbox_sandbox.py tests/test_engine_host_channel.py tests/test_hosting_worker_sandbox.py -q`
   - result: `65 passed`
52. `python -m pytest tests/test_engine_host_channel.py tests/test_hosting_toolbox_sandbox.py -q`
   - result: `58 passed`
53. `python -m pytest tests/test_hosting_toolbox_sandbox.py tests/test_engine_host_channel.py tests/test_hosting_worker_sandbox.py -q`
   - result: `67 passed`
54. `python -m pytest tests/test_engine_host_channel.py tests/test_hosting_toolbox_sandbox.py -q`
   - result: `59 passed`
55. `python -m pytest tests/test_hosting_toolbox_sandbox.py tests/test_engine_host_channel.py tests/test_hosting_worker_sandbox.py -q`
   - result: `68 passed`
56. `python -m pytest tests/test_engine_host_channel.py::test_toolbox_lifecycle_channel_methods_forward_expected_payloads tests/test_hosting_toolbox_sandbox.py::test_sandboxed_toolbox_facade_shapes_requests_for_host_api -q --basetemp=.tmp_pytest_receipt_nodes`
   - result: `2 passed`
57. `python -m pytest tests/test_hosting_toolbox_sandbox.py::test_environment_execute_install_records_simulated_success -q`
   - result: `1 passed`
58. `python -m pytest tests/test_hosting_toolbox_sandbox.py::test_environment_verify_receipt_detects_missing_locked_package -q`
   - result: `1 passed`
59. `python -m pytest tests/test_hosting_toolbox_sandbox.py -q`
   - result: `46 passed`
60. `python -m pytest tests/test_hosting_toolbox_sandbox.py tests/test_engine_host_channel.py tests/test_hosting_worker_sandbox.py -q`
   - result: `69 passed`
61. `python -m pytest tests/test_hosting_worker_sandbox.py::test_brokered_filesystem_denies_traversal_and_allows_root_scoped_io -q`
   - result: `1 passed`
62. `python -m pytest tests/test_hosting_worker_sandbox.py::test_service_brokered_filesystem_uses_registration_policy -q`
   - result: `1 passed`
63. `python -m pytest tests/test_hosting_worker_sandbox.py::test_spawn_persists_sandbox_policy_and_runtime -q`
   - result: `1 passed`
64. `python -m pytest tests/test_hosting_worker_sandbox.py::test_plain_launcher_uses_close_fds_when_parent_handles_disabled -q`
   - result: `1 passed`
65. `python -m pytest tests/test_engine_host_channel.py::test_toolbox_lifecycle_channel_methods_forward_expected_payloads tests/test_hosting_toolbox_sandbox.py::test_sandboxed_toolbox_facade_shapes_requests_for_host_api -q`
   - result: `2 passed`
66. `python -m pytest tests/test_hosting_toolbox_sandbox.py -k "gate_call_reports_denied_and_allowed or toolbox_gate_reports_denied_and_allowed_outcomes or hosted_gate_denial_before_execute or hosted_toolbox_ref_aliases_and_ref_style_methods_shape_requests or executes_request_tools_via_hosted_toolbox" -q`
   - result: `5 passed`
67. `python -m pytest tests/test_engine_host_channel.py::test_toolbox_lifecycle_channel_methods_forward_expected_payloads -q`
   - result: `1 passed`
68. `python -m pytest tests/test_mp13chat_hosted_toolbox_api.py tests/test_hosted_chat_demo.py -q`
   - result: `7 passed`
69. `python -m pytest tests/test_hosted_tool_visibility.py tests/test_mp13chat_hosted_toolbox_api.py tests/test_hosted_chat_demo.py -q`
   - result: `9 passed`
70. `python -m pytest tests/test_engine_host_channel.py::test_toolbox_lifecycle_channel_methods_forward_expected_payloads tests/test_hosting_toolbox_sandbox.py::test_toolbox_gc_reconciles_stale_registrations_and_artifacts -q`
   - result: `2 passed`
71. end-to-end hosted chat demo validation in the user environment:
   - baseline hosted prompts worked with no fallback attempt to `scriptable_calculator`
   - negative HTTP test produced: `PermissionError - brokered_http_url_not_allowed:https://example.org/`
   - negative filesystem traversal test produced: `BrokeredFsError - path_traversal_denied`
   - `/t` now showed effective hosted-aware availability:
     - `SimpleCalc`, `ProjectFilePeek`, `ExampleHttpPeek` -> `Yes / hosted`
     - local intrinsic tools -> `No / gated`
   - `/t sc` now showed the effective model-facing toolbox state:
     - `Advertised tools: ExampleHttpPeek, ProjectFilePeek, SimpleCalc`
     - `Hosted-visible tools: ExampleHttpPeek, ProjectFilePeek, SimpleCalc`
     - `Hosted-gated tools: scriptable_calculator, scriptable_calculator_guide, symbolic_algebra, symbolic_algebra_guide`
72. `python -m pytest tests/test_hosting_toolbox_sandbox.py::test_ensure_toolbox_assignments_ready_returns_rollout_metadata tests/test_hosting_toolbox_sandbox.py::test_wait_for_toolbox_executor_ready_requires_inventory_match -q`
   - result: `2 passed`
73. `python -m pytest tests/test_engine_host_channel.py::test_toolbox_lifecycle_channel_methods_forward_expected_payloads tests/test_hosting_toolbox_sandbox.py::test_toolbox_references_reports_referenced_and_stale_artifacts -q`
   - result: `2 passed`
74. `python -m pytest tests/test_engine_host_channel.py::test_toolbox_lifecycle_channel_methods_forward_expected_payloads tests/test_hosting_toolbox_sandbox.py::test_toolbox_consistency_reports_profile_registration_and_environment_mismatches -q`
   - result: `2 passed`
75. `python -m pytest tests/test_engine_host_channel.py::test_toolbox_lifecycle_channel_methods_forward_expected_payloads tests/test_hosting_toolbox_sandbox.py::test_toolbox_repair_rebuilds_inconsistent_toolbox_from_persisted_state -q`
   - result: `2 passed`
76. `python -m pytest tests/test_engine_host_channel.py::test_toolbox_lifecycle_channel_methods_forward_expected_payloads tests/test_hosting_toolbox_sandbox.py::test_toolbox_reconcile_chains_consistency_repair_and_gc -q`
   - result: `2 passed`
77. `python -m pytest tests/test_hosting_toolbox_sandbox.py::test_wait_for_toolbox_executor_ready_requires_inventory_match tests/test_hosting_toolbox_sandbox.py::test_toolbox_reconcile_chains_consistency_repair_and_gc tests/test_engine_host_channel.py::test_toolbox_lifecycle_channel_methods_forward_expected_payloads -q`
   - result: `3 passed`
78. `python -m pytest tests/test_toolbox_admin.py -q`
   - result: `4 passed`
79. `python -m pytest tests/test_engine_host_channel.py::test_toolbox_lifecycle_channel_methods_forward_expected_payloads tests/test_hosting_toolbox_sandbox.py::test_toolbox_review_snapshot_filters_and_recommends_reconcile tests/test_toolbox_admin.py -q`
   - result: `6 passed`
80. `python -m pytest tests/test_hosting_toolbox_sandbox.py -k "hosted_toolbox_ref_serializes_and_deserializes" -q`
   - result: `2 passed`
81. `python -m pytest tests/test_toolbox_admin.py -k real_hosted_demo_toolbox -q`
   - result: `1 passed`
   - this covers the admin review path against a real hosted-chat-demo toolbox shape with three persisted sandbox profiles, while mocking only the low-level worker-launch/readiness boundary for determinism

Covered by tests:

1. sandbox policy normalization
2. spawn persistence of sandbox policy and runtime metadata
3. plain launcher `close_fds` behavior
4. live Windows Low-IL denial of write to a medium-integrity file
5. live Windows named-pipe RPC continuity for a sandboxed helper worker
6. brokered filesystem root-scoped read/write/list and traversal denial
7. brokered HTTP allowlist enforcement and response shaping
8. worker-side brokered filesystem and HTTP client payload construction
9. control-channel broker helper payload forwarding
10. existing daemon/channel regression slices around startup and hosting auth/security
11. toolbox bundle manifest staging and registration metadata
12. host-side toolbox allowlist denial before worker dispatch
13. native toolbox async parallel execution in the new harness
14. harness round-robin scheduling across a sandbox executor pool
15. end-to-end `toolbox.describe` / `toolbox.execute` over the dedicated toolbox executor IPC worker
16. native toolbox gate decisions for allowed vs undefined tools
17. hosted toolbox gate routing for allowed vs denied tools
18. control-channel forwarding for `toolbox-gate`
19. hosted execution harness denial before sandbox dispatch when the gate rejects a call
20. hosted chat/router advertisement filtering metadata through router configuration and describe-based discovery
21. hosted router summary state for chat-facing diagnostics
22. effective hosted-aware tool visibility summarization for prompt/inspection use
23. end-to-end hosted chat demo usability in the user environment, including clean negative-path broker denials and effective hosted-aware toolbox inspection
24. explicit toolbox reconciliation sweep for stale executor registrations, stale bundle roots, and unreferenced environments
25. stricter rollout readiness validation using executor-reported tool inventory, not only toolbox IPC responsiveness
26. explicit reference reporting for referenced vs stale toolbox engines, bundle roots, and environment roots before reconciliation
16. direct `host.call` RPC on the toolbox executor worker
17. toolbox execution using brokered filesystem callback context (`context.fs.read_text(...)`)
18. toolbox-worker startup spec generation and worker manifest resolution via `MP13_TOOLBOX_WORKER_SPEC_PATH`
19. toolbox worker host-service metadata resolution via startup spec instead of only env vars
20. intrinsic-only staged toolbox revisions and manifest-driven intrinsic worker materialization
21. end-to-end sandbox execution of builtin intrinsic tools over the dedicated toolbox executor IPC worker
22. auto-discovered callable registration from staged modules and end-to-end sandbox execution of that path
23. host-side routing of tool execution across multiple sandbox-profile-specific executor registrations sharing one logical `toolbox_id`
24. automatic grouping of auto-callable requests by derived sandbox profile and routed spawn under one logical toolbox id
25. persistent logical toolbox state and incremental auto-registration updates with per-profile replacement rollout
26. unregister/removal lifecycle for persisted logical toolbox membership, including per-profile rebuild and full toolbox teardown
27. control-channel forwarding for high-level logical toolbox registration and removal payloads
28. deterministic toolbox environment metadata derivation and reuse across compatible revisions
29. host-created toolbox venv roots, worker command selection via venv Python, and environment-root GC when no longer referenced
30. readiness-gated toolbox rollout and rollback on failed warmup
31. simple toolbox-facing facade coverage for sandboxed auto-callable registration and execution
32. basic rollout metadata capture for successful profile cutovers
33. direct module-backed Python callable registration through the high-level facade
34. bounded rollout history persisted across successful profile replacements
35. intrinsic-tool registration and removal through high-level facade, service, and control surfaces
36. manual tool-definition registration and removal through high-level facade, service, and control surfaces
37. named environment-description persistence, linkage, and package-gap resolution
38. explicit environment-description apply operations that rebuild linked toolbox profiles and refresh environment metadata
39. named-environment inheritance across base-env chains, including lineage-aware package resolution and base-env apply propagation into derived environments
40. explicit environment-realization metadata planning and provenance recording for realized toolbox env roots
41. explicit environment-description sync from linked tool requirements, including update-in-place or clone-and-then-apply/realize flows
42. explicit environment install-plan emission, including generated requirements artifacts and pip command metadata without executing installs
43. explicit policy-gated environment install execution with persisted blocked/ok/failed result tracking
44. explicit install locking and stale-lock verification before execution

Important clarification:

1. these tests validate reusable sandbox plumbing and Windows enforcement slices
2. they now validate an initial dedicated toolbox sandbox executor lifecycle and IPC path
3. they now validate an initial generic toolbox-worker callback path
4. they now validate the structured toolbox-worker startup-spec path
5. they now validate startup-spec carriage of hosting-state metadata
6. they now validate deterministic `.venv` metadata derivation, real venv-root creation, worker spawn via the venv interpreter, reuse, and GC, but not locked dependency install lifecycle
7. they do not yet validate trustworthy direct-network route control on Windows
8. they no longer treat the trusted engine worker as a sandbox-executor integration target

### Not Yet Complete

1. bundle removal / executor reload / garbage-collection lifecycle for stale bundle revisions
2. immutable, host-built `.venv` lifecycle keyed by dependency profile, including locked dependency installation and stronger provenance rather than inheriting the base interpreter package set
3. richer rollout policies beyond the current readiness gate, such as replica warmup, longer-lived health history, and staged cutover
4. executor-side cancellation and richer broker callback contract completion
5. automatic daemon-managed pool lifecycle for sandbox executor replicas
6. Linux backend
7. any trustworthy claim of direct-network route enforcement on Windows
8. broader app/runtime adoption beyond the current selectable hosted execution path, lightweight execution router, and lightweight tool-round runtime helper

### New Architectural Clarification

The current implementation still assumes one toolbox executor registration corresponds to one staged sandbox revision.

That is good enough for:

1. one isolated sandbox profile per logical toolbox slice
2. equivalent executor replicas behind one harness

It is not yet the final model for:

1. one logical toolbox containing functions that require different permission/dependency specs
2. full lifecycle management including stronger garbage collection and richer rollout policy over time

Planned migration direction:

1. keep expanding the routing layer and orchestrator into richer persistent lifecycle management
2. move user-facing registration toward "callable + permissions spec + required imports" while hiding sandbox placement details
3. add higher-level app-facing facades on top of the new service/channel/CLI lifecycle methods
4. strengthen garbage collection beyond the current retired-bundle cleanup

### Assessment

The current repository state should be understood as:

1. sandbox groundwork exists
2. Windows starter enforcement exists
3. brokered I/O enforcement exists on the host side
4. an initial sandboxed toolbox executor path now exists with host-side authorization and manifest-driven loading
5. parallel toolbox execution is available via async dispatch, sandbox executor pools, or both
6. generic host callbacks now work for toolbox executors through `host.call` and convenience execution-context wrappers
7. toolbox startup can now move through a structured startup-spec path carrying manifest and hosting-state metadata rather than only loose env plumbing
8. legacy env wiring remains as a transitional fallback while startup-spec migration completes
9. native toolbox execution remains available when sandboxing is disabled or not desired
10. sandbox revisions can now represent builtin intrinsic tool activation in addition to staged user callables
11. host-side routing by `toolbox_id` can now dispatch tool calls across multiple sandbox-profile-specific executor registrations
12. host-side orchestration can now derive sandbox profiles from permissions/import specs and spawn grouped routed executors for one logical toolbox
13. host service can now persist logical toolbox membership and incrementally replace profile-specific executors as new auto-callables are registered
14. host service can now remove persisted logical toolbox membership, rebuild affected profiles, and tear down a logical toolbox when the last tool is removed
15. toolbox environment identity is now derived and persisted, and compatible revisions can reuse the same environment metadata root
16. toolbox environment roots are now created and reused by hosting, and unused roots are cleaned up when no logical toolbox profile still references them
17. logical toolbox registration updates now wait for replacement executors to become reachable before old registrations are retired
18. callers now have a simple facade path for common auto-callable sandbox registration and execution without manually building request payloads
19. successful profile cutovers now record basic rollout timing metadata in both API results and persisted toolbox profile state
20. callers can now register a real module-backed Python callable through the facade without manually constructing staged file/module payloads
21. persisted toolbox profile state now keeps a bounded history of successful rollouts rather than only the latest rollout snapshot
22. builtin intrinsic tools can now be managed through the same high-level sandboxed toolbox API and routed through the same hosted sandbox lifecycle
23. explicit manual tool definitions can now be managed through the same high-level sandboxed toolbox API and routed through the same hosted sandbox lifecycle
24. toolbox functions can now be linked to named environment descriptions, and hosting can report package gaps between linked functions and the selected environment
25. updated environment descriptions can now be explicitly applied to linked toolbox sandboxes, rebuilding the affected profiles and refreshing their realized environment metadata

That means the next phase is architectural rather than incremental:

1. complete bundle lifecycle management beyond initial staging
2. add `.venv` profile metadata and lifecycle keyed by intrinsic dependency tiers plus user dependencies
3. add callback contract refinement and optional cancellation
4. add daemon-managed pool orchestration and health/reload behavior
5. decide when to remove or reduce the remaining toolbox-worker compatibility env vars
