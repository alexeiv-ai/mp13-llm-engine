# Interactive Engine Host CLI Status and Plan

## Objective
Enhance `engine_host_cli.py` to support an interactive mode (`--interactive`) with a user-friendly terminal UI, similar to `hosting_config_cli.py`, but with a different color scheme.

## Phase 1: Interactive Menu Foundation
- [x] Parse `--interactive` argument in `engine_host_cli.py`.
- [x] Implement color scheme support (choose "light" or slightly modified tokens to differentiate from `hosting_config_cli.py`).
- [x] Implement `_prompt_menu`, `_c`, `_print_title`, `_print_block`, etc., matching the user experience of `hosting_config_cli.py`.

## Phase 2: Feature Implementation
- [x] **Feature 1:** List loaded engines and sandboxes with short details. Use CLI command logic (`discover-running`, `toolbox-describe`) or `EngineHostService` methods, format output to avoid raw JSON.
- [x] **Feature 2:** List more details of a specific engine or sandbox.
- [x] **Feature 3:** Print daemon process metrics (`host-metrics` logic).
- [x] **Feature 4:** List currently connected consumers and linked engines/sandboxes. Check `auth-list-sessions` or equivalent.
- [x] **Feature 5:** Kill an engine/sandbox or disconnect a consumer and kill linked resources (`shutdown` / `op-cancel` / `auth-revoke-session`).
- [x] **Feature 6:** Start/stop the daemon.

## Phase 3: Testing and Refinement
- [x] Run the interactive mode to test UI flow.
- [x] Ensure no raw JSON dumps occur.
- [x] Verify that error handling is smooth within the interactive mode.
- [x] Fix bug where consumer disconnect menu shows empty string or `Back` when sessions only contain prefixes.
- [x] Fix missing `get_pid` method error in `DaemonPidFile` when stopping daemon.
- [x] Show extra consumer details like TTL, connection time, offline stale warning, role, allowed scopes, and SSH binding.
- [x] Show token previews in consumer disconnect menu.
