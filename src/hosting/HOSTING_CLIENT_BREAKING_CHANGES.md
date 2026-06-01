# Hosting Client Breaking Changes And Migration Notes

Date: 2026-06-01

Purpose: track dependent-project changes required by the hosted sandbox runtime refactor. This file should be updated as implementation lands. Entries are written as client-facing stop/start guidance.

## Planned Migration: Workflow Python

- [x] Stop treating `workflow_python_helper` as the long-term primary API.
- [x] Start using `workflow_python` once available.
  - `workflow_python(profile=helper, environment_name=workflow-python-helper)` replaces the current helper lane.
  - `workflow_python(profile=node)` is planned for long-running workflow node execution with streaming responses.
  - Initial host surfaces now exist for helper-profile compatibility; dependent projects should wait for integration guidance before removing old helper calls.

- [ ] Stop routing workflow Python pools only by `engine_id`.
- [ ] Start accepting a host-derived `environment_key`.
  - The host will derive or verify the key from environment name, profile, Python runtime identity, imports, package pins or dependency lock identity, and sandbox policy hash.
  - Different environment keys will not share Python worker processes or hot child pools.

- [ ] Stop assuming `python.package_pins` are installed/enforced merely because they are present in an execute request.
- [x] Start using explicit workflow environment APIs for dependency-bearing environments.
  - Prepare.
  - Lock.
  - Verify.
  - Install when explicitly allowed.
  - Verify install receipt.
  - Execute only against a verified environment when dependencies matter.

- [ ] Stop using workflow execution code to install dependencies.
- [ ] Start treating dependency installation as a host-controlled environment-management operation.

- [ ] Stop relying on package ID or workflow ID to isolate process pools.
- [ ] Start using package/workflow IDs as provenance/audit fields unless they change runtime dependencies or policy.

- [ ] Stop assuming helper responses are the only workflow response shape.
- [ ] Start handling the new workflow response envelope.
  - `status` / `ok`.
  - output or helper result.
  - structured error.
  - runtime/environment metadata.
  - metrics.
  - audit metadata.
  - progress/log/artifact events for streaming profiles.

- [ ] Stop omitting `request_id` for cancelable or long-running work.
- [ ] Start passing stable `request_id` for request lifetime tracking and cancellation.

## Planned Migration: Workflow JS

- [ ] Stop treating `workflow_js_helper` as a separate long-term architecture.
- [ ] Start using `workflow_js(profile=helper)` once available.
- [ ] Start reading resources/capacity/cancellation state from the same environment-keyed resource model as workflow Python.
- [ ] Continue handling JS-specific compatibility fields until they are removed after migration.

## Planned Migration: Resources, Capacity, Metrics

- [ ] Stop reading capacity only from a single helper engine ID.
- [ ] Start reading capacity and resources by runtime kind and `environment_key`.
- [ ] Start consuming latency and concurrency metrics.
  - Queue wait ms.
  - Execution latency ms.
  - Total request lifetime ms.
  - Active calls.
  - Available slots.
  - Saturation count.
  - Timeout count.
  - Cancellation count.
  - Recent request outcomes.

- [ ] Stop assuming capacity changes apply globally to all workflow Python helpers.
- [ ] Start setting capacity per `environment_key`.

## Planned Migration: Streaming

- [ ] Stop using sync helper execution for long-running workflow node work.
- [ ] Start using streaming APIs for `workflow_python(profile=node)`.
  - Open stream.
  - Receive progress/log/artifact/result/error events.
  - Send cancel.
  - Close stream.

- [ ] Start tolerating partial progress and terminal events as separate records.

## Planned Migration: CLI And Interactive CLI

- [ ] Stop scripting only old commands after new workflow commands are available.
- [ ] Start using new workflow commands for new integrations.
- [ ] Old helper commands will remain temporary aliases during migration.
- [ ] Interactive CLI screens will move from helper-only views to workflow runtime pool views keyed by environment.

## Removal Candidates After Migration

- [ ] Remove or reduce `workflow_python_helper_ipc.py` to a thin compatibility entrypoint.
- [ ] Remove or reduce `workflow_js_helper_ipc.py` to a thin compatibility entrypoint.
- [ ] Remove old helper-specific service branches once dependent projects use workflow runtime APIs.
- [ ] Remove compatibility response fields only after clients confirm migration.

## Client Action Checklist

- [ ] Add client-side support for host-derived `environment_key`.
- [ ] Add client-side support for workflow runtime kind/profile fields.
- [ ] Add client-side support for streaming workflow node responses.
- [ ] Add client-side support for environment prepare/lock/verify/install flows.
- [ ] Add client-side support for per-environment resource/capacity views.
- [ ] Add client-side support for request lifetime and cancellation state.
- [ ] Keep schema validation and workflow authorization in the GUI/backend.
- [ ] Let host enforce runtime isolation, environment routing, and sandbox policy.
