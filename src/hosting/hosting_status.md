# Hosting Status: Model Worker Reconciliation Plan

## Context verified in this iteration

- `EngineHostService.connect_from_config()` currently creates a new engine id with `_next_engine_id()` for every successful connect request. If the base id already exists, it appends suffixes such as `_2` and `_3`.
- The current generated id can duplicate semantic content because it combines the config stem and model folder/name-derived text, e.g. `granite-2b_granite-3_3-2b-instruct_3`.
- `hosting.engine_worker_ipc` currently initializes one engine instance at worker startup from `MP13_ENGINE_ID`, `MP13_ENGINE_CONFIG_PATH`, and `MP13_MODEL_PATH`.
- The worker protocol currently supports `hello`, `rpc_call`, streaming, and HTTP compatibility. It does not expose explicit `load_model`, `unload_model`, or per-model config binding operations.
- The host registry currently treats `engine_id` as both the host-visible resource id and the worker/model instance identity. That makes automatic reuse of the same loaded model awkward and makes multi-model workers a breaking API/design change.
- Claim policy controls ownership of existing engine ids/resources, but it does not dedupe "same config + same model path" into an existing worker by default.
- Reachability is a daemon-to-worker `hello` probe. A live process with `reachable=false` is alive according to OS PID checks but did not answer the hosting worker IPC handshake.

## Small fixes completed

- [x] Add `require_auth`, key count, and session count fields to `host-metrics`.
- [x] Add `Auth: required/not required` to the interactive daemon status header when status metadata is available.
- [x] Derive resource state/kind in the interactive CLI even when connected to an already-running older daemon that does not yet return new `state`/`kind` fields.

## Proposed API outcome

- `worker_id` identifies a worker process.
- `engine_id` or `model_instance_id` identifies a loaded model/config binding inside a worker.
- `connect_from_config()` reconciles by default:
  - If a live reachable worker already has the same canonical model path and compatible runtime profile, reuse it.
  - If the requested config differs but the model path matches, add a new config binding/model instance record to the existing worker instead of launching a second process.
  - Return a structured outcome such as `status: "reused"` or `reconciled: true`, plus `worker_id`, `engine_id`/`model_instance_id`, and `config_binding_id`.
- Fresh process launch becomes explicit:
  - Add an explicit option such as `force_new_worker=True` or `launch_policy="fresh_worker"`.
  - Existing callers that need isolated model processes must opt in.
- Naming convention:
  - Worker id should be short and process-oriented, e.g. `worker-granite-2b-<hash>`.
  - Model/config instance id should be stable and semantic, e.g. `model-granite-3-3-2b-instruct-<hash>` or caller-supplied.
  - Avoid concatenating config name and model name unless both are needed and normalized once.

## Implementation checklist

- [x] Define registry schema migration:
  - Add `worker_id`, `loaded_models`, and `config_bindings` to registrations.
  - Preserve enough old fields to read existing `engine_id` registrations during migration or recovery.
  - Decide whether old `engine_id` is treated as both `worker_id` and default `model_instance_id`.
- [x] Extend worker protocol:
  - Add `model.list`, `model.load`, `model.unload`, and `model.describe` RPCs or equivalent hosting-private methods.
  - Decide whether `mp13_engine.mp13_engine_api` can support multiple initialized `instance_id` values in one process safely.
  - Ensure every inference request carries the target model/config binding.
- [x] Rework `connect_from_config()`:
  - Canonicalize model path and config path.
  - Search live reachable registrations for matching model path/runtime profile.
  - Reuse existing worker unless explicit fresh launch is requested.
  - Add a config binding when config differs.
  - Return whether it spawned, reused, or attached a config binding.
- [x] Rework proxy/routing APIs:
  - Route traffic by `engine_id`/`model_instance_id` to the owning `worker_id`.
  - Keep claim/token policy aligned with the externally visible model/config resource, not only the worker process.
- [x] Rework shutdown/unload semantics:
  - `unload_model` removes one model/config binding from a worker.
  - `shutdown worker` terminates the process and all loaded models.
  - Interactive CLI should offer both "Unload model" and "Stop worker" when multiple models share a worker.
- [x] Rework display:
  - List workers with PID/reachability.
  - Under each worker, list loaded model/config bindings.
  - Show stale/unreachable worker recovery options.
- [x] Update client docs after implementation:
  - `HOSTING_CLIENT_BREAKING_CHANGES.md` should describe the new default reuse behavior.
  - Clients that previously assumed `connect_from_config()` always spawned a fresh process must pass the explicit fresh-launch option.
  - Clients should store returned `worker_id` and `engine_id`/`model_instance_id` separately.
  - Clients should use unload APIs for model lifecycle rather than killing the whole worker when only one model binding should be removed.

## Troubleshooting unreachable workers

- Check the worker PID is alive.
- Check the registration `worker_ipc_family` and `worker_ipc_address`.
- Inspect the worker log path from the registration.
- Run `discover-running` and compare `alive`, `reachable`, and `reachability.error`.
- If `alive=true` and `reachable=false`, the current host can force-stop workers through local recovery. There is no model-level auto-heal yet because the daemon cannot know whether killing/restarting a worker is safe for all active consumers without the reconciliation/ownership model above.
