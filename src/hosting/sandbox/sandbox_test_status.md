# Sandbox Test Status

Date: 2026-04-04
Scope: what is currently tested, how to run it, and what the polished hosted-toolbox smoke flows should look like.

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

### 3.5 Admin/operator helper slice

```powershell
python -m pytest tests/test_toolbox_admin.py -q
```

## 4. Current Verified Results

Verified in the user environment:

1. `python -m pytest tests/test_hosting_toolbox_sandbox.py -q`
   - `46 passed`
2. `python -m pytest tests/test_hosting_toolbox_sandbox.py tests/test_engine_host_channel.py tests/test_hosting_worker_sandbox.py -q`
   - `69 passed`
3. `python -c "import app.mp13chat as m; print('ok', hasattr(m, '_handle_live_prompt'), hasattr(m, 'configure_hosted_toolbox_execution'))"`
   - `ok True True`

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
4. the remaining gaps are mainly hardening and breadth, not missing fundamentals

## 8. Still Worth Manual Attention

1. live dead-worker detection while registrations still exist
2. any environment-specific Windows pipe/ACL oddities
3. future Linux backend testing
