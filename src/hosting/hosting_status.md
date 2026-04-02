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
    - `SandboxedToolboxFacade`
    - wraps `toolbox_register_auto(...)`, `toolbox_unregister_auto(...)`, `toolbox_describe(...)`, and `toolbox_execute(...)`
    - hides low-level auto-request shaping for common sandboxed callable registration flows
29. Added first-slice rollout observability in [engine_host_service.py](/o:/repos/mp13-llm-engine/src/hosting/engine_host_service.py):
    - register/unregister now return `rollout` metadata for newly readied executors
    - persisted logical toolbox profile state now records basic rollout metadata such as `ready_at` and `warmup_ms`
30. Extended the higher-level toolbox facade in [toolbox_harness.py](/o:/repos/mp13-llm-engine/src/hosting/toolbox_harness.py):
    - `SandboxedToolboxFacade.register_python_callable(...)`
    - callers can now register a real module-backed Python callable without manually supplying staged file/module metadata
31. Added bounded per-profile rollout history in [engine_host_service.py](/o:/repos/mp13-llm-engine/src/hosting/engine_host_service.py):
    - successful `register_auto` / `unregister_auto` cutovers now append history entries with action, engine ids, bundle revision, and warmup timing
32. Extended high-level sandbox toolbox lifecycle to builtin intrinsics:
    - `EngineHostService.toolbox_register_intrinsics(...)`
    - `EngineHostService.toolbox_unregister_intrinsics(...)`
    - control channel / daemon / CLI forwarding for the same operations
    - `SandboxedToolboxFacade.register_intrinsic_tools(...)`
    - `SandboxedToolboxFacade.unregister_intrinsic_tools(...)`
33. Extended high-level sandbox toolbox lifecycle to explicit manual tool definitions:
    - `EngineHostService.toolbox_register_manual(...)`
    - `EngineHostService.toolbox_unregister_manual(...)`
    - control channel / daemon / CLI forwarding for the same operations
    - `SandboxedToolboxFacade.register_manual_tool(...)`
    - `SandboxedToolboxFacade.unregister_manual_tool(...)`

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
8. broader higher-level facade coverage and app integration beyond the current auto-callable/module-backed callable/intrinsic/manual-definition facade

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

That means the next phase is architectural rather than incremental:

1. complete bundle lifecycle management beyond initial staging
2. add `.venv` profile metadata and lifecycle keyed by intrinsic dependency tiers plus user dependencies
3. add callback contract refinement and optional cancellation
4. add daemon-managed pool orchestration and health/reload behavior
5. decide when to remove or reduce the remaining toolbox-worker compatibility env vars
