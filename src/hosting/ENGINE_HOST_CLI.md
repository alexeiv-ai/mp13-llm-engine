# Engine Host CLI

`hosting_cli.py` is the project-root wrapper for `python -m hosting.engine_host_cli`.
It can start local hosting daemons, relay remote daemon control over SSH, and issue
one-shot JSON control commands.

Top-level options must appear before the subcommand:

```powershell
py hosting_cli.py --ssh-target user@example-host --control-ssh-key C:\keys\id_ed25519 host-metrics
```

## Invocation Modes

`--interactive`
: Launches the terminal menu. Local targets use the local daemon lifecycle helpers.
Remote targets use `EngineHostControlChannel` and SSH relay control.

`<subcommand>`
: Runs one command and prints a JSON envelope: `{"ok": true, "result": ...}` or
`{"ok": false, "error": ...}`. If a remote target or client profile is supplied,
the command is sent through `EngineHostControlChannel`. Otherwise the CLI first
tries the local daemon and then falls back to direct local `EngineHostService`
behavior for compatibility.

`--daemon`, `--daemon --background`
: Starts the local daemon. These flags are parsed before normal argparse command
handling and are used by local lifecycle helpers.

`--relay-wrapper`
: Remote SSH entrypoint used by `EngineHostControlChannel`. It starts/locates the
remote local daemon according to remote policy, then relays stdin/stdout to the
remote daemon IPC. This is normally configured as the SSH forced command or used
through `--engine-host-remote-cmd`.

## Target Options

`--ssh-target`, `--remote-target`, `--engine-host-ssh-target`
: Selects SSH relay mode. Value is the SSH destination, for example
`user@example-host`. When present, non-interactive commands and interactive menu
actions use `EngineHostControlChannel` instead of local daemon/service fallback.

`--control-endpoint`
: Endpoint override. Values like `ssh://user@example-host` or `user@example-host`
also imply SSH relay mode.

`--control-ssh-key`
: Local private key file used by `ssh` to reach the remote host. Remote control is
non-interactive and expects key-based SSH authentication.

`--ssh-known-hosts-line`
: Pinned known-hosts line used for strict SSH host verification. This is required
for remote restart and recommended for all remote control.

`--engine-host-remote-cmd`
: Remote base command for the hosting CLI. Defaults to
`python -m hosting.engine_host_cli`. Relay connections use/append
`--relay-wrapper` as needed.

`--control-ssh-fingerprint`
: Expected control SSH key fingerprint. When sessions are issued with SSH binding,
this fingerprint is included in the binding metadata.

## Client-Realm Profile Options

`--client-profile`
: Loads a client-realm profile and lets `EngineHostControlChannel` resolve target,
SSH key, known-hosts, and related control settings.

`--client-realm`
: Realm name for profile and secret lookup. Defaults to `default` when omitted.

`--client-realm-root`
: Root folder for client-realm files. If omitted, the default client-realm
location is used by profile resolution.

`--client-secret-password`
: Password for materializing protected client-realm secrets when the selected
profile references a stored SSH private key.

## Local State Options

`--engines-state-file`
: Local managed-engine registry path. Used by local daemon startup and direct
local fallback.

`--control-state-file`
: Local hosting access/control state path. Used by local daemon startup, direct
local fallback, and local reset helpers.

`--pid-file`
: Local daemon PID file. Used to discover local daemon port, start local daemon
instances, and stop local daemons.

## Auth And Payload Options

`--session-token`
: Existing daemon session token to attach to control commands.

`--payload-json`
: Inline JSON payload for the selected subcommand.

`--payload-stdin`
: Reads the JSON payload from stdin.

Subcommands also accept selector flags:

`--engine-id`
: Adds `engine_id` to the command payload when provided.

`--resource-kind`, `--resource-id`
: Adds resource selectors to the command payload when provided.

## Remote Authentication Model

Remote control uses two separate credentials. They are easy to confuse, but they
are not interchangeable.

Transport key
: The SSH key passed with `--control-ssh-key` or resolved from `--client-profile`.
It must be accepted by the remote SSH server and normally should be installed as
a forced-command key for `python -m hosting.engine_host_cli --relay-wrapper`.
This key opens the encrypted relay transport only. It cannot issue daemon
sessions by itself.

Daemon auth key
: A hosting RBAC key registered in the remote daemon auth state, usually an
`admin` public-key auth entry. Its private key signs `auth-begin-challenge`
payloads and produces a daemon `session_token`. This key authorizes daemon
commands after the SSH relay is reachable.

If the remote host is configured for remote connectivity, the daemon also
requires SSH binding on issued sessions. `EngineHostControlChannel` supplies that
binding automatically from the SSH target and optional `--control-ssh-fingerprint`.
This means remote public-key authentication must happen through the remote
channel, not through a disconnected local service fallback.

If you only have the daemon admin private key but do not have a working transport
key/profile, you cannot control the remote daemon with this CLI. First import or
provision the transport profile/key, then authenticate to the daemon.

## Running With A Password-Protected Admin Key

The password-protected daemon admin key is used locally to sign the daemon
challenge. The remote daemon never receives the private key or its password.

Interactive flow with a complete key file:

1. Preflight the transport key:

   ```powershell
   ssh -T -o BatchMode=yes -i C:\keys\transport_ed25519 user@example-host python -m hosting.engine_host_cli --help
   ```

2. Start the remote interactive menu:

   ```powershell
   py hosting_cli.py --ssh-target user@example-host --control-ssh-key C:\keys\transport_ed25519 --ssh-known-hosts-line "example-host ssh-ed25519 AAAA..." --interactive
   ```

3. Choose an action. If the daemon returns `session_token_required`, the menu
   prompts for admin private-key material. Paste a file path such as
   `C:\keys\admin_ed25519`, or paste a client-realm `SecretRecord` JSON blob, or
   paste the raw OpenSSH private-key block.

4. If the admin key is passphrase-protected, `ssh-keygen` prompts on the local
   terminal for that key passphrase. This prompt is local; it is not a remote SSH
   password prompt and does not require a remote PTY.

Interactive flow with only the raw OpenSSH private-key block:

1. Start the menu with the same transport flags or client profile.
2. When prompted for the admin key, paste the full raw block:

   ```text
   -----BEGIN OPENSSH PRIVATE KEY-----
   ...
   -----END OPENSSH PRIVATE KEY-----
   ```

3. Submit an empty line after the block. If the key is passphrase-protected,
   answer the local `ssh-keygen` passphrase prompt.

For non-interactive commands, do not expect the CLI to prompt for the admin key
or passphrase. Use one of these patterns instead:

- First issue a session token with the public-key challenge flow, then pass that
  token to one-shot commands:

  ```powershell
  $begin = py hosting_cli.py --ssh-target user@example-host --control-ssh-key C:\keys\transport_ed25519 --ssh-known-hosts-line "example-host ssh-ed25519 AAAA..." --payload-json '{"key_id":"admin-main","scope":"control"}' auth-begin-challenge | ConvertFrom-Json
  $challenge = $begin.result.challenge
  $challengeId = $begin.result.challenge_id
  Set-Content -Path .\challenge.txt -Value $challenge -NoNewline
  ssh-keygen -Y sign -f C:\keys\admin_ed25519 -n engine-host-auth .\challenge.txt
  $signature = Get-Content .\challenge.txt.sig -Raw
  $completePayload = @{ challenge_id = $challengeId; signature_ssh = $signature } | ConvertTo-Json -Compress
  $complete = py hosting_cli.py --ssh-target user@example-host --control-ssh-key C:\keys\transport_ed25519 --ssh-known-hosts-line "example-host ssh-ed25519 AAAA..." --payload-json $completePayload auth-complete-challenge | ConvertFrom-Json
  $token = $complete.result.token
  ```

  `ssh-keygen` may prompt locally for the admin key passphrase in this sequence.
  After that, use the token:

  ```powershell
  py hosting_cli.py --ssh-target user@example-host --control-ssh-key C:\keys\transport_ed25519 --ssh-known-hosts-line "example-host ssh-ed25519 AAAA..." --session-token <token> host-metrics
  ```

- Use application code with `EngineHostControlChannel` plus
  `hosting.client_realm_api.begin_client_key_authentication()` and
  `complete_client_key_authentication()` if your application has its own secure
  passphrase prompt/signing layer.
- For unattended automation, use a pre-issued short-lived token or an
  environment-specific secret manager that can sign the challenge without a TTY.

## Display And Help

`--examples [COMMAND]`
: Prints command examples. With a command name, prints examples for that command.

`--color-scheme {dark,light}`
: Selects color tokens for interactive output.

`-h`, `--help`
: Prints argparse usage.

## Remote Examples

One-shot remote metrics:

```powershell
py hosting_cli.py --ssh-target user@example-host --control-ssh-key C:\keys\transport_ed25519 --ssh-known-hosts-line "example-host ssh-ed25519 AAAA..." --session-token <token> host-metrics
```

Interactive remote menu:

```powershell
py hosting_cli.py --ssh-target user@example-host --control-ssh-key C:\keys\transport_ed25519 --ssh-known-hosts-line "example-host ssh-ed25519 AAAA..." --interactive
```

Using a client profile:

```powershell
py hosting_cli.py --client-realm-root C:\mp13\client-realm --client-profile demo host-metrics
```

## Remote Lifecycle Notes

Normal remote commands use SSH relay mode and do not require a PTY on the remote
host. Remote daemon restart is not a daemon RPC; it is an out-of-band SSH exec of
the remote hosting CLI with `--daemon --background`. Because SSH runs with
`BatchMode=yes`, remote restart requires key-based SSH auth and cannot answer a
remote password prompt.

This has direct operator consequences:

- Do not expect `hosting_cli.py --ssh-target ... --interactive` to ask for the
  remote account password. It will fail instead of waiting for a password prompt.
- Do not rely on first-connect host-key prompts. Provide `--ssh-known-hosts-line`
  or use a client profile that includes pinned host-key material.
- Do not configure remote restart around `sudo` commands that require an
  interactive password. If the remote daemon needs elevated privileges, configure
  the remote host so the daemon can be started non-interactively by policy, for
  example through a service manager, a forced command, or narrowly scoped
  passwordless sudo outside this CLI.
- Do not point remote control at an SSH key that requires a passphrase unless the
  passphrase is already available to the local `ssh` process, for example through
  `ssh-agent`. The CLI will not open a remote PTY to collect it.

Before using remote CLI commands, preflight SSH from the same shell:

```powershell
ssh -T -o BatchMode=yes -i C:\keys\transport_ed25519 user@example-host python -m hosting.engine_host_cli --help
```

If that command prompts, remote control is not ready. Fix SSH key auth, host-key
pinning, agent setup, or remote command policy first. After the preflight works,
use either explicit CLI flags or a client-realm profile so callers do not have to
repeat key and host-pin settings.

`reset-hosting-access` is local-only and is rejected when a remote target/profile
is explicitly selected.
