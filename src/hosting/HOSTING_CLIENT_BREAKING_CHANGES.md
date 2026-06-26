# Hosting Client Breaking Changes

## Service-Broker Host API Registration

Known `fs.*` and `http.fetch` Host Capability descriptors now represent
daemon-owned brokered IO:

1. `provider.kind` is `service_broker`.
2. `provider.provider_id` is `builtin.service_broker` in descriptor/discovery
   metadata.
3. Descriptions and argument schemas are derived from the static service-broker
   registry docstrings/signatures.

Client action:

1. Use `EngineHostControlChannel.known_host_capability_methods(...)` only as
   service-broker descriptor metadata.
2. Use `host_capability_session_register_known_methods(...)` or
   `host_capability_session_register_service_broker_methods(...)` to expose
   daemon-owned `fs.*` / `http.fetch` for a request, instance, workflow, or
   consumer scope.
3. Do not expect `host_capability_session_register_known_methods(...)` to
   default to `provider_kind="client_session"` anymore. It now defaults to
   `provider_kind="service_broker"` and forces `binding.transport` to
   `service_broker`.
4. Register custom client/backend callable methods with
   `host_capability_session_register(..., provider_kind="client_session", ...)`.
5. Register hosted toolbox exported methods with `provider_kind="toolbox_session"`.

Approval behavior:

1. Service-broker calls run through the normal Host Capability approval path
   before daemon-local brokered IO executes.
2. `allow_once` and `add_to_scope` decisions work the same as for client
   provider sessions.
3. Approval does not widen sandbox policy. Filesystem roots, root access mode,
   brokered filesystem enablement, brokered HTTP enablement, network mode, host
   allowlists, URL prefixes, timeout, and response limits remain enforced by
   the daemon broker.

Node worker behavior:

1. Python and JS node workers discover selected service-broker methods through
   `host.describe()` / `api.describe()`.
2. `host.call("fs.read_text", {...})`, Python convenience wrappers, and JS
   `api.fs.*` wrappers call the same Host Capability route.
3. If no matching service-broker session is registered for the worker scope,
   `fs.*` and `http.fetch` are unsupported even if sandbox broker policy allows
   the underlying IO.

Toolbox status:

Toolbox brokered IO is not migrated in this slice. It still uses toolbox-native
dispatch and policy checks. A later phase will converge toolbox
`context.host.call`, `context.fs.*`, and `context.http.*` onto the shared
service-broker registry/dispatcher.
