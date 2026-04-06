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
13. hosted generic callback relay slices:
    - callback context propagation
    - callback concurrency
    - hosted execution-harness callback binding forwarding
14. brokered callback attribution slices:
    - brokered filesystem / HTTP service echo callback context
    - live toolbox execution proves brokered filesystem callback attribution to original tool call

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

### 3.4B Brokered callback attribution slices

```powershell
python -m pytest tests/test_hosting_worker_sandbox.py -q -k "brokered_filesystem or brokered_http"
python -m pytest tests/test_hosting_toolbox_sandbox.py -q -k "live_callback or context_fs_wrapper or host_call_rpc_uses_host_dispatch"
```

### 3.5 Admin/operator helper slice

```powershell
python -m pytest tests/test_toolbox_admin.py -q
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
     - `hosting.engine_host_service`
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
4. the remaining gaps are mainly hardening and breadth, not missing fundamentals

## 8. Still Worth Manual Attention

1. live dead-worker detection while registrations still exist
2. any environment-specific Windows pipe/ACL oddities
3. broader Linux distro/platform coverage beyond the currently validated WSL Ubuntu shadow setup
