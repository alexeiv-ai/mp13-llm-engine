# Hosting Client Breaking Changes

## `connect_from_config()` now reconciles by default

Hosting clients should no longer assume that every successful `connect_from_config()` call starts a new worker process.

Default behavior is now:

- The host canonicalizes the requested config path and model path.
- If a live, reachable model worker already has the same canonical model path and a compatible runtime profile, the host reuses that worker.
- If the config differs but the model path matches, the host adds a new config binding to the existing worker instead of launching a second process.
- The response reports the outcome with `spawned`, `reconciled`, and a `status` such as `ok`, `attached`, or `reused`.

Why: loading the same model repeatedly wastes memory and startup time. The host now treats the worker process and the externally visible model/config binding as separate resources.

## Same model, different configs

Different configs that resolve to the same canonical model path can coexist in one worker process. In that case, the host creates separate config bindings, but they point at the same loaded `model_instance_id`.

This means:

- Each config binding gets its own client-visible `engine_id` and `config_binding_id`.
- Claims, tokens, proxy calls, and lifecycle tracking still operate on the client-visible `engine_id`.
- The worker does not load a separate copy of the model for each config binding.
- Config values that require a different loaded model runtime, GPU isolation, process-level environment, or incompatible startup profile require a fresh worker.

Use `force_new_worker=True` or `launch_policy: "fresh_worker"` when a config must be isolated from another config for the same model.

## Store worker and model identifiers separately

Consumers should persist these response fields as distinct concepts:

- `worker_id`: the worker process identity. Use this for process lifecycle operations such as stopping the whole worker.
- `engine_id`: the externally visible model/config binding id. Use this for claims, tokens, proxy calls, and normal inference routing.
- `model_instance_id`: the loaded engine instance inside the worker. This may be shared by multiple config bindings.
- `config_binding_id`: the host registry binding between a config and a loaded model instance.

Older registrations are migrated on read. For old rows, the previous `engine_id` is treated as both the `worker_id` and default `model_instance_id`.

## Opt in when you need a fresh process

Clients that require process isolation must request it explicitly:

```python
host.connect_from_config(
    config_path="default",
    model_path="C:/models/granite",
    force_new_worker=True,
)
```

Equivalent payload form:

```json
{
  "config_path": "default",
  "model_path": "C:/models/granite",
  "launch_policy": "fresh_worker"
}
```

Use this only when process isolation is required, for example when testing runtime startup, isolating GPU memory ownership, or intentionally running separate model workers for the same model.

## Use unload for model lifecycle

Do not stop the whole worker when the intent is to remove one model/config binding. Use `unload_model(engine_id)` or the `unload-model` command for model lifecycle cleanup.

Use worker shutdown only when every loaded model/config binding in that worker should be terminated.

## Routing and claims

Continue using `engine_id` for:

- claim and token APIs
- proxy HTTP requests
- proxy RPC and streaming APIs
- client-visible model/config resource ownership

The host routes that `engine_id` to the owning `worker_id` and target `model_instance_id` internally. Inference requests sent through the proxy now carry the target engine instance to the worker so shared workers do not rely on ambient defaults.
