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

The interactive menu also contains a local-only recovery/auth submenu. Those
actions intentionally bypass daemon RPC and read or mutate local control state
through `EngineHostService`; they are not available for remote targets.

Interactive menu capabilities include:

- Listing loaded model workers, generic/tool workers, sandbox metadata,
  reachability, process CPU/RAM, and worker-reported GPU VRAM when available.
- Loading a model from a hosted config store entry. If the config does not name a
  model path, the menu asks for one. Load progress is displayed from daemon
  operation status and worker load progress. The loader reuses a compatible
  already-loaded model worker by default, and also reuses reachable idle model
  workers rather than starting a second process unless `Force new engine
  instance` is selected.
- Unloading model bindings through the engine worker management API. The menu
  waits for completion and verifies daemon discovery no longer reports the
  selected model binding.
- Stopping workers/sandboxes and revoking auth sessions.
- Listing live consumer connections separately from issued auth sessions. The
  live view marks the current interactive CLI connection.
- Testing a loaded model by sending a prompt through hosting `proxy-rpc-call` to
  `run-inference`. The CLI prints response text separately from metrics and
  compares model-reported generation duration with CLI-observed end-to-end
  latency.

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

## Session Validation Command

`auth-validate-session`
: Validates a token the caller already holds for a requested scope and optional
expected key id. For SSH targets, the channel supplies the current SSH binding
so callers can verify the token is valid for this connection before adopting or
reusing it.

## Hosting Status Commands

`hosting-setup-status`
: Returns sanitized hosting setup metadata, including configured mode, auth
posture, lifecycle policy, key/session counts, and secure-state summary. Use
this instead of having GUI/backend integrations read hosting-owned files such as
`access_control.json`, `bootstrap_state.json`, or `client_key_map.json`
directly.

`hosting-secure-state-status`
: Returns metadata-only secure-state status for hosting-owned files. It reports
missing/plaintext/encrypted/locked state and the documented startup env names,
but it does not decrypt or return file contents. Current daemon-owned encrypted
state reads are intentionally disabled until daemon startup key propagation is
wired.

## Workflow Runtime Commands

Workflow Python helper-profile workers are managed through workflow runtime facades:

```powershell
@'{"profile":"helper","environment_name":"workflow-python-helper","capacity":2,"session_token":"<control_token>"}'@ | py hosting_cli.py --payload-stdin workflow-python-ensure
@'{"profile":"helper","environment_key":"<environment_key>","session_token":"<control_token>"}'@ | py hosting_cli.py --payload-stdin workflow-python-resources
@'{"profile":"helper","environment_key":"<environment_key>","capacity":4,"session_token":"<control_token>"}'@ | py hosting_cli.py --payload-stdin workflow-python-set-capacity
@'{"profile":"helper","environment_key":"<environment_key>","request_id":"req-1","session_token":"<control_token>"}'@ | py hosting_cli.py --payload-stdin workflow-python-cancel-request
```

Workflow JavaScript uses the QuickJS-backed node facade with
`workflow-js-ensure`, `workflow-js-execute`, `workflow-js-resources`,
`workflow-js-set-capacity`, and `workflow-js-cancel-request`. JS requests use
`profile:"node"` and do not accept Node.js runtime selection such as
`node_executable`.

The interactive menu exposes workflow runtime resource, capacity, refresh,
request-status, stream receive, and request-cancel actions under `Manage
workflow runtimes`.

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

3. If the daemon requires auth, choose `Authenticate` from the main menu before
   protected actions. Protected actions fail with an authenticate-first message
   instead of starting an auth prompt implicitly. For SSH targets, paste a file
   path such as `C:\keys\admin_ed25519`, a client-realm `SecretRecord` JSON blob,
   or the raw OpenSSH private-key block.

4. If the admin key is passphrase-protected, `ssh-keygen` prompts on the local
   terminal for that key passphrase. This prompt is local; it is not a remote SSH
   password prompt and does not require a remote PTY.

For local-only shared-secret deployments, the interactive `Authenticate` menu
also offers `Shared key password`, which issues a control session via
`auth-issue-session`. That option is intentionally unavailable for SSH targets
and remote-capable access profiles.

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

  To check a token before reuse:

  ```powershell
  py hosting_cli.py --ssh-target user@example-host --control-ssh-key C:\keys\transport_ed25519 --ssh-known-hosts-line "example-host ssh-ed25519 AAAA..." --payload-json '{"token":"<token>","scope":"control","expected_key_id":"admin-main","check_ssh_binding":true}' auth-validate-session
  ```

- Use application code with `EngineHostControlChannel` plus
  `hosting.client_realm_api.begin_client_key_authentication()` and
  `complete_client_key_authentication()` if your application has its own secure
  passphrase prompt/signing layer.
- For unattended automation, use a pre-issued short-lived token or an
  environment-specific secret manager that can sign the challenge without a TTY.

## Local Recovery/Auth Tools

The interactive CLI has a `Local recovery/auth tools` submenu for cases where the
local daemon is stopped, unreachable, or has auth state that needs repair. This
submenu is intentionally local-only. It uses the configured `--control-state-file`
and `--engines-state-file` directly and does not use the daemon control channel.

Use it for local recovery, not for normal daemon operation. Remote targets must
use the SSH relay/channel path instead.

Available local recovery actions:

`Show local auth status`
: Reads local control state and prints auth status such as `require_auth`, key
count, session count, challenge count, and roles.

`Authenticate locally with admin private key`
: Uses local `EngineHostService` to create and complete a public-key challenge in
the local control state. The private-key prompt accepts a file path, a
client-realm `SecretRecord` JSON blob, or a raw OpenSSH private-key block. If the
admin key is passphrase-protected, `ssh-keygen` prompts locally for that
passphrase. The resulting session token is kept in the current interactive CLI
process and can make stopped-daemon offline reads work until the token expires.

`List local sessions`
: Reads saved sessions directly from local control state. This may include stale
or expired entries after pruning rules run. Normal remote-capable consumers
should validate a token they already hold with `auth-validate-session`; session
listing is metadata-only and does not return bearer tokens.

`Revoke local session`
: Shows a numbered list of saved local sessions, then mutates local control
state to revoke the selected token preview. This is useful when a token should
be invalidated before restarting the daemon.

`List local auth keys`
: Reads registered key IDs, roles, auth methods, and disabled state from local
control state.

`Revoke local auth key`
: Shows a numbered list of registered local keys, then mutates local control
state to revoke the selected key. This requires an explicit confirmation prompt.

`Clear local auth keys/sessions`
: Stops or terminates the local daemon if possible, then clears only saved auth
keys, active sessions, and pending auth challenges from local control state. It
does not reset hosting to unconfigured: access policy such as `require_auth`,
endpoint mode, lifecycle profile, setup artifacts, keyring/bootstrap files, and
client-realm custody are kept. If `require_auth=true`, this can leave the host
requiring auth with no registered keys until setup or RBAC repair adds a key.

`Force stop daemon and workers`
: Stops registered workers from local state, scans for local orphan
`hosting.engine_worker_ipc` processes, then terminates the local daemon PID if it
is still alive. Use this when the daemon is unreachable or stale and normal stop
cannot run the daemon shutdown path.

`Force restart daemon and workers`
: Runs the force-stop action, then starts a fresh local daemon. Use this when an
old daemon PID is alive but local control is unreachable and blocks startup.

Daemon start uses lifecycle policy when an old PID is alive but local control is
unreachable:

- `exclusive` endpoint mode or `foreground_terminal_bound` lifecycle: the CLI
  treats the daemon as recoverable local state, force-stops workers/daemon, then
  starts a fresh daemon.
- `shared` endpoint mode with `detached_user_process` lifecycle: the CLI does
  not kill it automatically because it may be serving another long-lived
  consumer. Use `Force restart daemon and workers` when the operator has decided
  that process is stale.

In all cases, only one daemon instance is allowed for a PID file. A second daemon
is not spawned on top of a live unreachable PID.

`Start daemon after recovery`
: Calls the normal local daemon bootstrap helper after recovery edits.

Offline read views in the normal menu are narrower: they can read local state
while the daemon is stopped when auth policy allows it or the interactive process
has a valid session token. If auth is required and no token is available, the
view starts the local admin-key auth workflow and retries the read after a token
is acquired. You can also authenticate first through `Local recovery/auth tools
-> Authenticate locally with admin private key`, or start the daemon and
authenticate through the normal daemon challenge flow.

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

`reset-hosting-access`, `force-stop-daemon`, and `force-restart-daemon` are
local-only and are rejected when a remote target/profile is explicitly selected.
`reset-hosting-access` is the auth-table clear helper described above; it is not
the full `hosting_config_cli` reset-to-unconfigured workflow.
