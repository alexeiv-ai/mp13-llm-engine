# Hosting Client Breaking Changes

Date: 2026-05-22

This document is intentionally reset for the workflow Python helper executor work. Complete it when the implementation is done, then send the relevant sections to dependent projects.

## Draft Reply To Dependent Project

Status: planned, not implemented yet.

We agree with the request. The previous hosting docs deferred a Python helper executor because there was no concrete caller contract at the time. Your new contract supplies that missing requirement, so the right next step is to add a dedicated hosted `workflow_python_helper` lane rather than continuing backend-owned local `subprocess.run(...)` execution.

The planned shape is:

- `spawn_workflow_python_helper(engine_id="workflow-python-helper")`
- `execute_workflow_python_helper` over hosting IPC/RPC
- `workflow_python_helper_resources(...)`
- `set_workflow_python_helper_capacity(...)`
- `cancel_workflow_python_helper_request(...)`
- `executor_kind = "workflow_python_helper"`
- `sandbox.profile = "workflow_python_helper_v1"`
- shared runtime environment manager reuse for `python.import_allowlist`, `python.package_pins`, and `python.environment_name`

We will also add generic helper pool aliases to JS helper resources while preserving existing JS-specific fields. That should let backend code use one resource/capacity/cancel controller for both JS and Python helper workers.

## Stop Doing

To be completed after implementation.

- [ ] Stop running workflow Python helpers locally with backend `subprocess.run(...)`.
- [ ] Stop generating local runner files as the execution boundary for approved workflow Python helpers.
- [ ] Stop passing caller-provided Python file paths as executable inputs to hosting.
- [ ] Stop treating Python helper execution as generic Python execution.
- [ ] Stop routing workflow Python helpers through toolbox state unless they are intentionally exposed as toolbox tools.

## Start Doing

To be completed after implementation.

- [ ] Spawn or discover workers registered with `executor_kind = "workflow_python_helper"`.
- [ ] Execute approved Python helpers with `proxy_rpc_call(..., method="execute_workflow_python_helper", ...)`.
- [ ] Send `module_source` plus `module_sha256`.
- [ ] Treat `source_path` as provenance only.
- [ ] Include stable `request_id` when cancellation or status correlation is needed.
- [ ] Send `python.import_allowlist`, `python.package_pins`, and `python.environment_name` when a realized runtime environment is needed.
- [ ] Use `workflow_python_helper_resources(...)` to inspect capacity, active request ids, process counts, PIDs, and metrics.
- [ ] Use `set_workflow_python_helper_capacity(...)` to resize a loaded helper worker.
- [ ] Use `cancel_workflow_python_helper_request(...)` to terminate a specific stuck request.
- [ ] Use the normalized `pool` resource shape for both JS and Python helper controls.

## Planned Python Example

```python
channel.spawn_workflow_python_helper(engine_id="workflow-python-helper")

channel.proxy_rpc_call(
    engine_id="workflow-python-helper",
    method="execute_workflow_python_helper",
    params={
        "module_source": source_text,
        "module_sha256": sha256_hex,
        "package_id": package_id,
        "workflow_id": workflow_id,
        "package_source_digest": package_source_digest,
        "source_path": source_path,
        "request_id": request_id,
        "export_name": "condition",
        "operation": "condition",
        "payload": payload,
        "provenance": provenance,
        "limits": {"timeout_ms": 5000, "output_limit_bytes": 65536, "memory_limit_mb": 128},
        "python": {
            "import_allowlist": import_allowlist,
            "package_pins": package_pins,
            "environment_name": "workflow-python-helper",
        },
    },
)

resources = channel.workflow_python_helper_resources(engine_id="workflow-python-helper")
channel.set_workflow_python_helper_capacity(engine_id="workflow-python-helper", capacity=2)
channel.cancel_workflow_python_helper_request(engine_id="workflow-python-helper", request_id=request_id)
```

## Planned Normalized Resource Shape

```json
{
  "capacity": 4,
  "active_calls": 1,
  "available_slots": 3,
  "pool": {
    "process_count": 4,
    "active_process_count": 1,
    "idle_process_count": 3,
    "active_request_ids": ["req-123"],
    "processes": [
      {
        "pid": 12345,
        "alive": true,
        "busy": true,
        "active_request_id": "req-123",
        "request_count": 7,
        "max_requests": 256,
        "reusable": true,
        "resources": {
          "cpu_percent": 1.2,
          "memory_mb": 64.0
        }
      }
    ]
  }
}
```

## Authorization Boundary

To be verified after implementation.

- [ ] `workflow_python_helper_resources`: `diagnostic_user` and above.
- [ ] `spawn_workflow_python_helper`: `worker_user`, `config_editor`, and `admin`.
- [ ] `set_workflow_python_helper_capacity`: `worker_user`, `config_editor`, and `admin`.
- [ ] `cancel_workflow_python_helper_request`: `worker_user`, `config_editor`, and `admin`.
- [ ] `execute_workflow_python_helper`: traffic-scoped proxy for the registered helper `engine_id`.
- [ ] Raw generic `spawn`: still `config_editor` and `admin` only.

## Compatibility Window

To be completed after implementation.

- [ ] Document whether backend local Python helper execution should be removed immediately or kept behind a short migration flag.
- [ ] Document stable Python helper `workflow_sandbox_*` failure reasons.
- [ ] Document any platform-specific limits for memory enforcement and child process metrics.
- [ ] Document JS helper resource compatibility fields that remain available after normalized `pool` aliases are added.
