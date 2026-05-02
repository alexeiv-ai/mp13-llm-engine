# Hosting Client Breaking Changes

This note covers client-facing changes around daemon control, client-realm key
custody, and public-key authentication helpers.

## Must Replace

Custom daemon routing code must be replaced with `EngineHostControlChannel`.

Do not instantiate `LocalSocketConnection` directly from GUI clients, scripts, or
interactive tools unless you are implementing the channel itself. Do not duplicate
command maps with `EngineHostService` fallback for runtime/config/auth/toolbox
commands. That bypasses the channel's target resolution, SSH relay support,
session handling, and command fallback policy.

Use:

```python
from hosting.engine_host_channel import EngineHostControlChannel

client = EngineHostControlChannel({
    "engine_host_client_realm_root": "...",
    "engine_host_client_profile": "demo",
})
metrics = client.get_host_metrics()
```

Or for ad hoc command dispatch:

```python
result = client.invoke_control_command("host-metrics", {})
```

Local-only service fallback is still appropriate for explicitly local recovery
helpers such as `reset_hosting_access`; it is not the normal client API.

## Remote Control

Clients that need to target a remote daemon should use either a client-realm
profile or explicit channel settings:

```python
client = EngineHostControlChannel({
    "engine_host_ssh_target": "user@example-host",
    "control_ssh_key": "C:/keys/id_ed25519",
    "ssh_known_hosts_line": "example-host ssh-ed25519 AAAA...",
})
```

The channel chooses `SSHRelayConnection` for SSH targets and
`LocalSocketConnection` for local targets. Remote relay does not require a PTY.
It expects non-interactive SSH key authentication.

Remote clients need two separate credentials:

- transport key/profile: opens the SSH relay to `--relay-wrapper`
- daemon auth key/session: signs daemon auth challenges or supplies an existing
  session token

Having only the daemon admin private key is not enough to control a remote
daemon. The transport key/profile must work first. Having only the transport key
is also not enough when `require_auth=true`; the daemon still requires a valid
session token or public-key challenge completion.

For remote challenge authentication, clients must let `EngineHostControlChannel`
attach SSH binding metadata. Use `client.auth_begin_challenge(...)`,
`begin_client_key_authentication(client, ...)`, `authenticate_client_with_key(...)`,
or `client.invoke_control_command("auth-begin-challenge", ...)`. Do not call a
local `EngineHostService.auth_begin_challenge(...)` for a remote daemon, and do
not manually send raw `auth-begin-challenge` payloads over a custom transport
unless you also include the correct `ssh_binding`.

## CLI Migration

`hosting_cli.py` now accepts remote/channel target flags before the subcommand:

```powershell
py hosting_cli.py --ssh-target user@example-host --control-ssh-key C:\keys\id_ed25519 host-metrics
py hosting_cli.py --client-realm-root C:\mp13\client-realm --client-profile demo --interactive
```

When `--ssh-target`, `--control-endpoint`, `--client-profile`, or
`--client-realm-root` is supplied, one-shot commands are routed through
`EngineHostControlChannel`. Without those flags, the existing local behavior is
preserved: try local daemon first, then direct local fallback.

`reset-hosting-access` is local-only and is rejected for explicit remote/profile
targets.

## Client-Realm Auth Helpers

The old interrupted helper shape mixed daemon challenge orchestration, terminal
prompting, and `ssh-keygen` signing. GUI and headless clients should use the new
step helpers from `hosting.client_realm_api` instead.

Use this for GUI/native signing flows:

```python
from hosting.client_realm_api import (
    begin_client_key_authentication,
    complete_client_key_authentication,
)

challenge = begin_client_key_authentication(client, key_id="admin-main")
signature = gui_key_store.sign(challenge["challenge_text"])
session = complete_client_key_authentication(
    client,
    challenge_id=challenge["challenge_id"],
    signature_ssh=signature,
)
token = session["token"]
```

Use this for a single orchestration call with a GUI signer callback:

```python
from hosting.client_realm_api import authenticate_client_with_key

token = authenticate_client_with_key(
    client,
    "admin-main",
    signer=lambda challenge: gui_key_store.sign(challenge["challenge_text"]),
)
```

Use this only for headless unencrypted OpenSSH private-key material:

```python
from hosting.client_realm_api import sign_client_auth_challenge_with_private_key

signature = sign_client_auth_challenge_with_private_key(
    private_key_text=private_key_text,
    challenge_text=challenge["challenge_text"],
)
```

## Must Not Do

Do not call `input()` inside reusable auth helpers.

Do not run `ssh-keygen` in reusable GUI/headless helpers in a mode that can block
on a terminal or askpass prompt. The provided `sign_client_auth_challenge_with_private_key`
helper is deliberately non-interactive and disables askpass.

Do not assume a remote target can be controlled with local PID-file logic. Remote
commands must go through `EngineHostControlChannel` and SSH relay settings.

## Should Replace

Existing UI code that manually maps commands such as `auth-list-sessions`,
`auth-revoke-session`, `auth-begin-challenge`, `auth-complete-challenge`, and
`host-metrics` should be simplified to call typed `EngineHostControlChannel`
methods where possible.

Existing scripts that accept local daemon PID/control-state flags should also
consider accepting the same remote/profile settings as `hosting_cli.py`:

- `engine_host_ssh_target`
- `control_ssh_key`
- `ssh_known_hosts_line`
- `engine_host_remote_cmd`
- `engine_host_client_profile`
- `engine_host_client_realm`
- `engine_host_client_realm_root`

This makes the script work for local and remote daemons through the same client
surface.

## Compatibility Notes

`authenticate_client_with_key(...)` still exists, but its intended use is now
orchestration. Prefer passing `signer=` for GUI clients. Passing
`private_key_text=` uses non-interactive `ssh-keygen` signing and requires
unencrypted key material.

`EngineHostControlChannel.invoke_control_command(...)` is the public escape hatch
for command names that do not yet have typed channel methods.
