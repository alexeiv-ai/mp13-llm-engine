# Sandbox Test Status

Date: 2026-04-04
Scope: what is currently tested, how to run it, and what the polished hosted-toolbox smoke flows should look like.

Update: 2026-04-05
The new gated-tool semantic slices are passing.
Update: 2026-04-06
The Windows generic hosted callback relay regression is fixed.
Generic hosted callback relay slices now pass on both native Windows and WSL/Linux validation paths.
Update: 2026-04-06
The first interactive hosted approval slice is implemented and passing.
Hosted approval now uses the callback processor with `tool_requires_confirmation` and decision values `deny`, `allow_once`, and `add_to_scope`.
Update: 2026-04-06
Hosted approval follow-up behavior is now covered too.
Repeated gated calls dedupe by tool name for sticky decisions, approval timeout defaults to deny, and the hosted runtime auto-forwards the active cursor plus `toolbox_ref` into callback context.
Update: 2026-04-06
Guide-policy first-slice behavior is now covered.
User guides resolve as first-class guide tools, do not inherit parent-tool gating implicitly, and can still be gated explicitly by guide name.
Update: 2026-04-06
Guide hardening is now in place too.
All guides execute through the static guide runner; intrinsic guides are now registered from static guide content rather than callable guide implementations.
Update: 2026-06-01
The hosted sandbox runtime refactor plan now treats this file as the test-navigation baseline. Existing helper characterization tests live in `tests/test_workflow_python_helper_ipc.py`, `tests/test_workflow_js_helper_ipc.py`, `tests/test_workflow_helper_service.py`, and workflow helper slices in `tests/test_engine_host_channel.py`. New shared runtime-base tests should be added without removing the current sandbox/toolbox/generic slices until the migration phases explicitly mark old helper implementations removable.
Update: 2026-06-01
Shared runtime, pool, Python environment, workflow Python facade, and CLI compatibility slices now have focused coverage:
`tests/test_hosting_sandbox_runtime_base.py`, `tests/test_hosting_sandbox_runtime_pool.py`, `tests/test_hosting_python_runtime_base.py`, `tests/test_workflow_python_contract.py`, `tests/test_workflow_helper_service.py`, `tests/test_engine_host_channel.py`, `tests/test_engine_host_cli_remote_args.py`, and `tests/test_engine_host_cli_interactive.py`.
Update: 2026-06-01
The refactor coverage now also includes internal process/JS runtime bases, runtime-env GC, workflow Python node stream command rollout, and auth/policy coverage:
`tests/test_hosting_sandbox_process_base.py`, `tests/test_hosting_js_runtime_base.py`, `tests/test_hosting_python_runtime_base.py`, `tests/test_workflow_helper_service.py`, `tests/test_engine_host_channel.py`, and `tests/test_hosting_auth_roles.py`.
Update: 2026-06-01
Phase 9 toolbox migration has initial coverage for shared hosted environment identity on toolbox registrations while preserving `toolbox_venvs`:
`tests/test_hosting_toolbox_sandbox.py -k "toolbox_runtime_base or orchestrator_spawn_uses_shared_environment_identity"`.

## 1. Environment

Run from repo root.

Windows PowerShell:

```powershell
$env:PYTHONPATH='src'
```

## 2. Core Automated Test Coverage

Current automated coverage includes:

1. sandbox policy normalization
2. sandbox launcher persistence and Windows Low-IL slices
3. brokered filesystem and brokered HTTP enforcement
4. dedicated toolbox executor IPC
5. host-side toolbox describe/execute/gate paths
6. logical toolbox routing across multiple sandbox profiles
7. persistent register/unregister lifecycle
8. named environment description lifecycle
9. install plan / lock / resolve / receipt plumbing
10. operator surfaces:
    - references
    - consistency
    - review snapshot
    - repair
    - reconcile
    - gc
11. hosted chat/tool visibility slices
12. hosted demo and admin review flow
13. hosted generic callback relay slices:
    - callback context propagation
    - callback concurrency
    - hosted execution-harness callback binding forwarding
14. hosted gated approval slices:
    - allow-once execution override
    - add-to-scope mutation for future calls in the same request
    - hosted ref approval path
    - timeout -> deny behavior
    - per-round dedupe for sticky approval decisions
    - hosted runtime auto scope-target forwarding
15. guide-policy slices:
    - user guides resolve as first-class guide names
    - gating a parent tool does not implicitly gate its guide
    - guides can still be explicitly gated by guide name
16. brokered callback attribution slices:
    - brokered filesystem / HTTP service echo callback context
    - live toolbox execution proves brokered filesystem callback attribution to original tool call
17. workflow helper characterization slices:
    - Python helper module identity, operation allowlist, timeout, output limit, cancellation, capacity, import allowlist, audit/provenance, child process reuse, and real round trip
    - JS helper module identity, operation allowlist, timeout, output limit, cancellation, capacity, audit/provenance, child process reuse, and real/service round trips where Node is available
    - service/channel/daemon helper spawn, resources, capacity, cancel, and runtime-environment realization paths
18. hosted sandbox runtime refactor slices:
    - deterministic runtime/environment key models and sandbox policy hashes
    - internal process base request lifecycle, stream session, cancellation, and status plumbing
    - process pool registry scheduling, capacity, cancellation, recent request, and metrics rollups
    - workflow-facing Python runtime environment prepare/lock/verify/install/receipt/select behavior
    - workflow-facing Python runtime env GC for unreferenced `runtime_envs`
    - JS runtime base identity and process-pool capability
    - `workflow_python(profile=helper)` service/channel/daemon/direct CLI facade coverage
    - `workflow_python(profile=node)` request/response contract and structured pending-worker envelope
    - `workflow_python(profile=node)` stream-open/recv/send/close rollout coverage with pending-worker events
    - environment-key pool isolation for incompatible policy/dependency identity
    - interactive CLI workflow pool view compatibility for annotated Python helper registrations
    - initial toolbox executor registration identity mapping through `HostedToolboxRuntimeBase`

## 3. Main Test Commands

### 3.1 Sandbox + hosting worker slices

```powershell
python -m pytest tests/test_hosting_worker_sandbox.py tests/test_hosting_worker_sandbox_windows_live.py -q
```

### 3.2 Toolbox sandbox slices

```powershell
python -m pytest tests/test_hosting_toolbox_sandbox.py -q
```

### 3.3 Broader hosted-toolbox regression slice

```powershell
python -m pytest tests/test_hosting_toolbox_sandbox.py tests/test_engine_host_channel.py tests/test_hosting_worker_sandbox.py -q
```

### 3.4 Hosted chat / visibility slices

```powershell
python -m pytest tests/test_mp13chat_hosted_toolbox_api.py tests/test_hosted_chat_demo.py tests/test_hosted_tool_visibility.py -q
```

### 3.4A Hosted callback slices

```powershell
python -m pytest tests/test_hosting_toolbox_sandbox.py -q -k "generic_callback or callbacks_run_concurrently or forwards_callback_processor"
```

Current note:

1. native Windows callback relay slices now pass again
2. WSL/Linux callback relay slices also pass
3. current Windows fix shape:
   - the per-execute callback relay still uses local `AF_PIPE`
   - callback relay named pipes are created up front with a low-integrity security descriptor so low-IL sandbox workers can connect back to the hosted caller relay
4. brokered callback attribution slices continue to pass

### 3.4B Brokered callback attribution slices

```powershell
python -m pytest tests/test_hosting_worker_sandbox.py -q -k "brokered_filesystem or brokered_http"
python -m pytest tests/test_hosting_toolbox_sandbox.py -q -k "live_callback or context_fs_wrapper or host_call_rpc_uses_host_dispatch"
```

### 3.5 Admin/operator helper slice

```powershell
python -m pytest tests/test_toolbox_admin.py -q
```

### 3.5A Workflow helper slices

```powershell
python -m pytest tests/test_workflow_python_helper_ipc.py tests/test_workflow_js_helper_ipc.py tests/test_workflow_helper_service.py tests/test_engine_host_channel.py -q -k "workflow"
```

### 3.5B Hosted runtime refactor slices

```powershell
python -m pytest tests/test_hosting_sandbox_runtime_base.py tests/test_hosting_sandbox_runtime_pool.py tests/test_hosting_python_runtime_base.py tests/test_workflow_python_contract.py tests/test_workflow_helper_service.py tests/test_engine_host_channel.py tests/test_engine_host_cli_remote_args.py tests/test_engine_host_cli_interactive.py -q -k "workflow or runtime or cli or interactive or contract"
```

Current expanded refactor validation command:

```powershell
python -m pytest tests/test_hosting_sandbox_runtime_base.py tests/test_hosting_sandbox_process_base.py tests/test_hosting_sandbox_runtime_pool.py tests/test_hosting_python_runtime_base.py tests/test_hosting_js_runtime_base.py tests/test_workflow_python_contract.py tests/test_workflow_helper_service.py tests/test_engine_host_channel.py tests/test_engine_host_cli_remote_args.py tests/test_engine_host_cli_interactive.py tests/test_hosting_auth_roles.py -q -k "workflow or runtime or process_base or js_runtime or cli or interactive or contract or worker_user or diagnostic_user"
```

### 3.6 WSL Ubuntu shared-shadow validation slice

Recommended model:

1. keep the main Windows checkout unchanged
2. create a WSL-native shadow root such as `~/mp13-wsl`
3. symlink shared code/content from the Windows checkout into that shadow root
4. keep the Linux `.venv` and Linux `poetry.lock` inside the WSL shadow root

Minimal example:

```bash
mkdir -p ~/mp13-wsl
cd ~/mp13-wsl

ln -s /mnt/o/repos/mp13-llm-engine/src src
ln -s /mnt/o/repos/mp13-llm-engine/tests tests
ln -s /mnt/o/repos/mp13-llm-engine/misc misc
ln -s /mnt/o/repos/mp13-llm-engine/pyproject.toml pyproject.toml
ln -s /mnt/o/repos/mp13-llm-engine/README.md README.md
ln -s /mnt/o/repos/mp13-llm-engine/mp13chat.py mp13chat.py
ln -s /mnt/o/repos/mp13-llm-engine/mp13config.py mp13config.py
ln -s /mnt/o/repos/mp13-llm-engine/configs configs
```

Then create the Linux env:

```bash
cd ~/mp13-wsl
cp /mnt/o/repos/mp13-llm-engine/poetry.lock poetry.lock
poetry config virtualenvs.in-project true --local
poetry install --with dev
```

If the copied lock turns out to be incompatible with Linux, regenerate it in the shadow root:

```bash
cd ~/mp13-wsl
rm -f poetry.lock
poetry lock --no-update
poetry install --with dev
```

Before running tests, validate the shadow root:

```bash
cd ~/mp13-wsl
python3 misc/wsl_shared_test_setup.py check
python3 misc/wsl_shared_test_setup.py commands
```

Main Linux validation commands:

```bash
cd ~/mp13-wsl
PYTHONPATH=src poetry run pytest tests/test_hosting_daemon_pidfile.py -q
PYTHONPATH=src poetry run pytest tests/test_hosting_toolbox_sandbox.py -q -k 'startup_spec or spec_path or spec_hosting or toolbox_executor_ipc_end_to_end'
PYTHONPATH=src poetry run pytest tests/test_engine_host_channel.py -q
```

Broader Linux backend slice:

```bash
cd ~/mp13-wsl
PYTHONPATH=src poetry run pytest tests/test_hosting_toolbox_sandbox.py tests/test_engine_host_channel.py tests/test_hosting_daemon_pidfile.py -q
```

## 4. Current Verified Results

Verified in the user environment:

1. `python -m pytest tests/test_hosting_toolbox_sandbox.py -q`
   - `46 passed`
2. `python -m pytest tests/test_hosting_toolbox_sandbox.py tests/test_engine_host_channel.py tests/test_hosting_worker_sandbox.py -q`
   - `69 passed`
3. `python -c "import app.mp13chat as m; print('ok', hasattr(m, '_handle_live_prompt'), hasattr(m, 'configure_hosted_toolbox_execution'))"`
   - `ok True True`

Verified from this session about WSL:

4. shared-shadow root at `~/mp13-wsl`
   - Poetry env path: `/home/alx/mp13-wsl/.venv`
   - import check passed for:
     - `pydantic`
     - `pytest`
     - `hosting.service.host_service`
5. `PYTHONPATH=src poetry run pytest tests/test_hosting_daemon_pidfile.py -q`
   - `19 passed`
6. `PYTHONPATH=src poetry run pytest tests/test_hosting_toolbox_sandbox.py -q -k 'startup_spec or spec_path or spec_hosting or toolbox_executor_ipc_end_to_end'`
   - `8 passed`
7. `PYTHONPATH=src poetry run pytest tests/test_hosting_toolbox_sandbox.py tests/test_engine_host_channel.py tests/test_hosting_daemon_pidfile.py -q`
   - `123 passed`
8. `python -m pytest tests/test_hosting_toolbox_sandbox.py -q -k "generic_callback or callbacks_run_concurrently or forwards_callback_processor"`
   - `3 passed`
9. `PYTHONPATH=src poetry run pytest tests/test_hosting_toolbox_sandbox.py -q -k 'generic_callback or callbacks_run_concurrently or forwards_callback_processor'`
   - `3 passed`
10. `python -m pytest tests/test_mp13chat_hosted_toolbox_api.py -q`
   - `14 passed`
11. `PYTHONPATH=src poetry run pytest tests/test_mp13chat_hosted_toolbox_api.py -q`
   - `14 passed`
12. `python -m pytest tests/test_hosting_worker_sandbox.py -q -k "brokered_filesystem or brokered_http"`
   - `6 passed`
13. `python -m pytest tests/test_hosting_toolbox_sandbox.py -q -k "live_callback or context_fs_wrapper or host_call_rpc_uses_host_dispatch"`
   - `2 passed`
14. `PYTHONPATH=src poetry run pytest tests/test_hosting_worker_sandbox.py -q -k 'brokered_filesystem or brokered_http'`
   - `6 passed`
15. `PYTHONPATH=src poetry run pytest tests/test_hosting_toolbox_sandbox.py -q -k 'live_callback or context_fs_wrapper or host_call_rpc_uses_host_dispatch'`
   - `2 passed`
16. `pytest -q tests/test_hosted_tool_visibility.py`
   - `7 passed`
17. `pytest -q tests/test_hosting_toolbox_sandbox.py -k "gated or blocked_in_scope or tools_view"`
   - `6 passed`
18. `pytest -q tests/test_mp13chat_hosted_toolbox_api.py -k "hosted_visible_hidden_and_gated_states or blocked_in_scope"`
   - `1 passed`
19. `pytest -q tests/test_hosted_tool_visibility.py tests/test_hosting_toolbox_sandbox.py -k "not test_hosted_toolbox_execute_routes_generic_callback_with_context and not test_hosted_toolbox_callbacks_run_concurrently"`
   - `101 passed, 2 deselected`
20. `pytest -vv tests/test_hosting_toolbox_sandbox.py -k "test_hosted_toolbox_execute_routes_generic_callback_with_context or test_hosted_toolbox_callbacks_run_concurrently"`
   - `2 passed, 94 deselected`
21. `pytest -q tests/test_hosting_toolbox_sandbox.py -k "generic_callback or callbacks_run_concurrently or forwards_callback_processor"`
   - `3 passed, 93 deselected`
22. `pytest -q tests/test_hosted_tool_visibility.py tests/test_hosting_toolbox_sandbox.py tests/test_mp13chat_hosted_toolbox_api.py`
   - `118 passed`
23. `PYTHONPATH=src poetry run pytest tests/test_hosting_toolbox_sandbox.py -q -k 'test_hosted_toolbox_execute_routes_generic_callback_with_context or test_hosted_toolbox_callbacks_run_concurrently'`
   - `2 passed, 94 deselected`
24. `pytest -q tests/test_hosting_toolbox_sandbox.py -k "approval or gated_requires_confirmation or callbacks_run_concurrently or forwards_tools_view or hosted_toolbox_ref_execute_approval"`
   - `7 passed, 92 deselected`
25. `pytest -q tests/test_hosted_tool_visibility.py tests/test_hosting_toolbox_sandbox.py tests/test_mp13chat_hosted_toolbox_api.py`
   - `121 passed`
26. `pytest -q tests/test_hosting_toolbox_sandbox.py -k "approval or timeout or hosted_toolbox_ref_execute_approval or facade_shapes_requests or aliases_and_ref_style"`
   - `6 passed, 94 deselected`
27. `pytest -q tests/test_mp13chat_hosted_toolbox_api.py -k "forwards_callback_processor or auto_forwards_scope_target or blocked_in_scope"`
   - `2 passed, 14 deselected`
28. `pytest -q tests/test_hosted_tool_visibility.py tests/test_hosting_toolbox_sandbox.py tests/test_mp13chat_hosted_toolbox_api.py`
   - `123 passed`
29. `python -m pytest tests/test_hosting_toolbox_sandbox.py -q`
   - `122 passed`

## 5. Polished Hosted Chat Smoke Flow

Launch hosted demo chat:

```powershell
$env:PYTHONPATH='src'
python -m app.mp13chat --hosted-demo --hosted-demo-toolbox-id toolbox-admin-demo --hosted-demo-project-root . --hosted-demo-hosting-root .tmp_toolbox_admin_demo
```

While chat is running, verify:

1. `compute 12 + 3 * 5`
2. `Use ProjectFilePeek to read src/app/mp13chat.py and show the first 300 characters.`
3. `Use ExampleHttpPeek to fetch https://example.com/ and show the first 200 characters.`
4. `/t`
5. `/t sc`

Negative-path prompts:

1. `Use ExampleHttpPeek to fetch https://example.org/ and show the first 100 characters.`
2. `Use ProjectFilePeek to read ../pyproject.toml`

Expected negative errors:

1. `PermissionError - brokered_http_url_not_allowed:https://example.org/`
2. `BrokeredFsError - path_traversal_denied`

Expected healthy chat/tool visibility:

1. advertised hosted tools:
   - `SimpleCalc`
   - `ProjectFilePeek`
   - `ExampleHttpPeek`
2. `/t` shows hosted tools as available via `hosted`
3. `/t sc` shows only hosted-visible tools under `Advertised tools`
4. local intrinsics can appear as hosted-gated

Attach `mp13chat` to an existing hosted toolbox instead of provisioning the demo:

```powershell
$env:PYTHONPATH='src'
python -m app.mp13chat --hosted-toolbox-id toolbox-admin-demo --hosted-engines-state-file .tmp_toolbox_admin_demo\managed_engines.json --hosted-control-state-file .tmp_toolbox_admin_demo\access_control.json
```

Expected attach behavior:

1. startup prints `Hosted toolbox attached.`
2. printed summary includes:
   - `toolbox_id`
   - `engines_state_file`
   - `control_state_file`
3. if hosted describe succeeds, advertised hosted tool names are printed from the hosted backend summary rather than hardcoded demo data

Thin-wrapper attach path:

```python
from app.hosted_toolbox_api import attach_existing_hosted_toolbox

attached = attach_existing_hosted_toolbox(
    toolbox_id="toolbox-admin-demo",
    engines_state_file=".tmp_toolbox_admin_demo/managed_engines.json",
    control_state_file=".tmp_toolbox_admin_demo/access_control.json",
)

assert attached.summary.get("mode") == "sandbox"
```

Minimal sample wrapper:

```powershell
$env:PYTHONPATH='src'
python demo/demo_hosted_toolbox_attach.py --toolbox-id toolbox-admin-demo --engines-state-file .tmp_toolbox_admin_demo\managed_engines.json --control-state-file .tmp_toolbox_admin_demo\access_control.json
```

Optional single tool execution:

```powershell
$env:PYTHONPATH='src'
python demo/demo_hosted_toolbox_attach.py --toolbox-id toolbox-admin-demo --engines-state-file .tmp_toolbox_admin_demo\managed_engines.json --control-state-file .tmp_toolbox_admin_demo\access_control.json --tool-name SimpleCalc --tool-arguments "{\"expr\":\"12 + 3 * 5\"}"
```

## 6. Polished Operator Smoke Flow

In a second terminal while hosted demo chat is still running:

### 6.1 Review snapshot

```powershell
$env:PYTHONPATH='src'
'{"toolbox_ids":["toolbox-admin-demo"]}' | python -m hosting.engine_host_cli --engines-state-file .tmp_toolbox_admin_demo\managed_engines.json --control-state-file .tmp_toolbox_admin_demo\access_control.json --payload-stdin toolbox-review-snapshot
```

Expected healthy result:

1. one toolbox
2. three profiles
3. zero issues
4. `recommended_action: "observe"`

### 6.2 Repair

```powershell
$env:PYTHONPATH='src'
'{"toolbox_ids":["toolbox-admin-demo"]}' | python -m hosting.engine_host_cli --engines-state-file .tmp_toolbox_admin_demo\managed_engines.json --control-state-file .tmp_toolbox_admin_demo\access_control.json --payload-stdin toolbox-repair
```

Expected healthy result:

1. `requested_toolbox_ids: ["toolbox-admin-demo"]`
2. `target_toolbox_ids: []`
3. `repaired_toolbox_ids: []`
4. `outcome: "noop"`

### 6.3 Reconcile

```powershell
$env:PYTHONPATH='src'
'{"toolbox_ids":["toolbox-admin-demo"]}' | python -m hosting.engine_host_cli --engines-state-file .tmp_toolbox_admin_demo\managed_engines.json --control-state-file .tmp_toolbox_admin_demo\access_control.json --payload-stdin toolbox-reconcile
```

Expected healthy result:

1. `target_toolbox_ids: []`
2. `repaired_toolbox_ids: []`
3. `outcome: "noop"`

### 6.4 Deep details only when needed

```powershell
$env:PYTHONPATH='src'
'{"toolbox_ids":["toolbox-admin-demo"],"details":true}' | python -m hosting.engine_host_cli --engines-state-file .tmp_toolbox_admin_demo\managed_engines.json --control-state-file .tmp_toolbox_admin_demo\access_control.json --payload-stdin toolbox-repair
```

```powershell
$env:PYTHONPATH='src'
'{"toolbox_ids":["toolbox-admin-demo"],"details":true}' | python -m hosting.engine_host_cli --engines-state-file .tmp_toolbox_admin_demo\managed_engines.json --control-state-file .tmp_toolbox_admin_demo\access_control.json --payload-stdin toolbox-reconcile
```

## 7. Interpretation

Current polished test story means:

1. the hosted toolbox core is working
2. the chat-hosted demo usability path is working
3. the operator/admin path is working with compact defaults
4. the new gated-tool Phase 1 semantics are working in native, hosted, and presentation slices
5. generic hosted callbacks are working again on both native Windows and WSL/Linux validation paths
6. the remaining gaps are mainly hardening and breadth rather than a known callback transport regression

## 8. Still Worth Manual Attention

1. live dead-worker detection while registrations still exist
2. broader Linux distro/platform coverage beyond the currently validated WSL Ubuntu shadow setup
3. additional Windows hardening around local IPC security descriptors is still worth periodic validation even though the current generic hosted callback relay regression is fixed
