# Hosting client breaking changes

## Daemon shutdown lifecycle diagnostics

The daemon PID file now separates process ownership from client availability.
Clients must not treat PID-file presence as "daemon is usable".

### New daemon status contract

Use `EngineHostControlChannel.get_daemon_status()` for lifecycle checks.

Relevant fields:

- `pid_alive`: the daemon process still exists.
- `reachable`: the daemon accepts normal control requests.
- `lifecycle_state`: `running`, `shutting_down`, or absent when no pid file is present.
- `reachability_error`: `daemon_shutting_down` when the pid file belongs to a daemon that is stopping.
- `shutdown_diagnostics`: structured shutdown progress and stale-pid investigation details.

When `lifecycle_state == "shutting_down"`:

- Do not open a normal daemon connection.
- Do not delete the pid file.
- Do not start a second daemon against the same pid file.
- Show a "stopping" state and poll `get_daemon_status()`.
- If the shutdown age exceeds the client UX timeout, offer an explicit force-recovery action.

### `stop_daemon()` response

`stop_daemon()` now returns post-stop daemon status:

```json
{
  "status": "shutdown_sent",
  "daemon_status": {
    "pid_alive": true,
    "reachable": false,
    "lifecycle_state": "shutting_down",
    "reachability_error": "daemon_shutting_down",
    "shutdown_diagnostics": {}
  }
}
```

Clients should close cached daemon connections after `shutdown_sent` and then poll
`daemon_status` / `get_daemon_status()` until `pid_alive == false` or the pid file
disappears.

### Bootstrap/start behavior

`bootstrap_daemon()` now blocks on a live daemon that is still shutting down and
returns:

```json
{
  "blocked_by_shutting_down_pid": true,
  "error": "existing daemon PID is still shutting down",
  "shutdown_diagnostics": {}
}
```

Client behavior:

- Back off and poll; do not start another daemon.
- Surface `shutdown_diagnostics.shutdown_stage`,
  `shutdown_diagnostics.shutdown_age_seconds`, and
  `shutdown_diagnostics.daemon_report_path`.
- Offer force recovery only from local operator/admin flows.

### Force recovery

Use `force_stop_daemon()` or `force_restart_daemon()` only as explicit recovery.
These helpers are local-host recovery helpers; they inspect and terminate local
daemon/worker processes. Remote clients must invoke an equivalent remote-side
helper rather than trying to kill a process from the client machine.

`force_stop_daemon()` now returns:

- `daemon_status_before_force`
- `stuck_shutdown`
- `worker_shutdown`
- `graceful_stop`
- `daemon_terminate`
- `daemon_status`

Treat force recovery as a bug-investigation event. Attach the returned object and
the daemon crash report to bug reports.

### Shutdown diagnostics

`shutdown_diagnostics` may include:

- `shutdown_requested_at`
- `shutdown_age_seconds`
- `shutdown_reason`
- `shutdown_requested_by`
- `shutdown_progress_updated_at`
- `shutdown_progress_age_seconds`
- `shutdown_stage`
- `shutdown_stage_status`
- `shutdown_stage_message`
- `operation_drain`
- `shutdown_checkpoints`
- `shutdown_stages`
- `daemon_report_path`

### Daemon crash report

The daemon diagnostic file remains:

```text
<hosting_root>/logs/daemon-crash.log
```

It is not a bounded multi-report history. By default it tracks one current
diagnostic report because `write_daemon_report(..., overwrite=True)` replaces the
file. Some lifecycle events, such as daemon startup, append separator-delimited
reports with `overwrite=False`; those appended entries are opportunistic context,
not a retention guarantee.

During shutdown, the daemon now writes `daemon_shutdown_progress` reports to this
file as stages advance. If shutdown gets stuck, the file should contain the last
known shutdown stage and relevant details for stale-pid investigation.
