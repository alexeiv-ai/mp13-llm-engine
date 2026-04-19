# Toolbox Controller Refactoring Plan and Status

## Status

Implemented.

-   `src/app/session_cursor_toolbox.py` was added as the app-level bridge between `ChatCursor`/`EngineSession` and toolbox scope utilities.
-   `src/app/session_toolbox_controller.py` was added as the UI-neutral command controller.
-   `src/app/mp13chat_tools_cli.py` was added as the CLI parser/renderer layer.
-   `src/app/tools_cli_light.py` was removed.
-   `src/app/mp13chat.py` now delegates toolbox command handling to the CLI layer and no longer carries duplicate toolbox scope helpers.
-   `src/app/engine_session.py` now owns the shared effective stack replay helpers, including stack-aware tools scope and adapter entry collection.
-   The session module does not depend on `context_cursor.py`.

Latest verification:

```text
poetry run python -m py_compile src\app\engine_session.py src\app\session_cursor_toolbox.py src\app\mp13chat.py
poetry run pytest tests\test_toolbox_controller.py tests\test_mp13chat_hosted_toolbox_api.py tests\test_hosting_toolbox_sandbox.py
# 141 passed

poetry run pytest
# 325 passed, 1 skipped
```

## Motivation

Before this refactor, the `LightweightToolsCliHandler` inside `src/app/tools_cli_light.py` mixed three distinct responsibilities:
1.  **Core Session/Cursor Tool Logic:** Navigating the `ChatCursor` and `Session` history to compute effective `ToolsScope` stacks, parsing wildcards, and normalizing tool names.
2.  **Toolbox Business Logic:** Executing commands to modify the `Toolbox` state (activate, deactivate, hide, show, add, modify, fix, replace).
3.  **Presentation & I/O (CLI):** Parsing user command strings, printing formatted tables with ANSI colors, and using `prompt_toolkit` to asynchronously ask for user input.

Furthermore, `src/app/mp13chat.py` contained redundant copies of several core logic functions (e.g., `_parse_scope_cli_args`, `_collect_tools_scope_entries`), violating DRY principles.

As other applications (APIs, GUIs, or automated agents) adopt the `ChatCursor` and `Session` architecture, they will need the core tool scoping and management logic *without* the CLI-specific presentation or I/O constraints.

## Goal

Refactor the tool management architecture to decouple session/toolbox bridge logic, command execution, and CLI presentation. The implemented architecture introduces a generalized `SessionToolboxController` that can be driven by non-CLI callers through injected interaction callbacks and structured result data.

Crucially, **the core `mp13_engine` must remain pure**. Concepts like `ChatCursor`, `Turn`, and `Session` are part of the app layer (`src/app/`), so any logic that bridges the core `Toolbox` with these session objects must also live in the app layer.

## Target Architecture

The refactoring divides the functionality into three focused layers within the app layer:

### 1. The Bridge (`src/app/session_cursor_toolbox.py`)
This module holds the pure logic that connects the core `Toolbox` to the app-level `ChatCursor` and `Session` objects. It does not handle CLI command parsing, ANSI coloring, prompt handling, or table formatting.
-   `collect_tools_scope_entries(cursor: ChatCursor) -> List[Tuple[Optional[str], ToolsScope]]`
-   `tool_wildcard_groups(toolbox: Toolbox) -> Dict[str, List[str]]`
-   `normalize_scope_tool_names(scope: ToolsScope, toolbox: Toolbox) -> Tuple[ToolsScope, List[str]]`
-   `all_tool_names(toolbox: Toolbox) -> List[str]`
-   `collect_tools_scope_entries` now delegates to `EngineSession.get_effective_tools_scope_entries(...)`, so stack-aware scope reconstruction is centralized in session code.
-   `EngineSession.get_effective_adapter_entries(...)` provides the analogous stack-aware adapter entry data used by `mp13chat.py`.

### 2. The CLI Parser (`src/app/mp13chat_tools_cli.py`)
This module owns CLI-only parsing and rendering. The parser layer can be tested without a live terminal.
-   Parses raw command strings and aliases (e.g., `m 1`, `scope set mode=silent`).
-   Parses CLI-specific argument syntaxes such as scope key/value arguments, pop target options, comma-separated target lists, numeric selections, and wildcard target expansion.
-   Converts controller result data into `mp13chat` terminal output, including ANSI colors and ASCII tables.
-   Wraps `prompt_toolkit` prompts and other CLI input functions.

### 3. The Controller (`src/app/session_toolbox_controller.py`)
A class (`SessionToolboxController`) responsible for applying parsed tool operations to the `Toolbox`, `ToolBoxRef`, and `ChatCursor`. It must **NOT** contain any `print()`, `input()`, ANSI color, or `prompt_toolkit` dependencies.
-   **State:** Holds references to the active `Toolbox`, `ToolBoxRef`, and callbacks to retrieve current status (like hosted tool summary).
-   **Result Surface:** Prefer returning structured status/messages/listing data from command methods. Lightweight output callbacks are acceptable where a direct stream of messages keeps the port smaller, but messages should be plain text and severity-tagged so the CLI decides formatting.
-   **I/O Abstraction:** Accepts callback functions only for interactions that truly require external UI:
    -   `prompt_user_fn(prompt_text: str) -> Awaitable[str]` -> For simple synchronous or asynchronous input (e.g., asking for choice 1, 2, or 3 during a `fix`).
    -   `prompt_multiline_fn(prompt_text: str) -> Awaitable[str]` -> For gathering JSON definitions during `replace`.
    -   `interactive_edit_fn(tool_name: Optional[str], edit_context: Dict[str, Any]) -> Awaitable[Tuple[bool, str]]` -> For launching an external editor (or similar UI).
    -   `get_external_tool_handler_fn() -> Callable` -> Provides the handler for interactive/external tools.
-   **Methods:** Exposes strongly-typed methods for each action, such as `cmd_enum()`, `cmd_modify(target: str)`, `cmd_scope(action: str, scope_args: str)`, `cmd_fix(target: str)`, etc.

### 4. External Tool Handler Ownership
Interactive external tool handling is CLI-specific, but `mp13chat.py` also needs an `external_tool_handler` symbol during toolbox load/reload. The implementation lives in `mp13chat_tools_cli.py`, with `mp13chat.py` retaining the compatibility path needed by existing startup and load flows.

## CLI Wrapper Details
-   It parses the raw CLI string (e.g., `m 1` or `scope set mode=silent`).
-   It instantiates the `SessionToolboxController`, wiring up its own CLI-specific I/O callbacks:
    -   `prompt_user_fn` / `prompt_multiline_fn`: Wraps `pt_session.prompt_async()`.
-   It delegates the actual execution to the controller's methods.
-   It will also house CLI-specific handlers currently stuck in `mp13chat.py`, such as the tools help renderer and the default external tool handler implementation.

## Execution Plan (Step-by-Step)

### Phase 1: Establish the Bridge (`session_cursor_toolbox.py`) - Done
1.  Created `src/app/session_cursor_toolbox.py`.
2.  Moved `collect_tools_scope_entries`, `normalize_scope_tool_names`, `tool_wildcard_groups`, and `all_tool_names` out of the old CLI handler.
3.  Kept bridge warnings as plain strings, not colorized terminal output.
4.  Removed duplicate toolbox scope collection from `src/app/mp13chat.py` once call sites used the new modules.
5.  Updated `collect_tools_scope_entries` to delegate stack-aware effective scope collection to `EngineSession`.

### Phase 2: Implement the `SessionToolboxController` - Done
1.  Created `src/app/session_toolbox_controller.py`.
2.  Defined the `SessionToolboxController` class and typed callbacks for required I/O.
3.  Ported the core business logic from `LightweightToolsCliHandler.handle_tools_command` into controller methods.
4.  Replaced hardcoded terminal I/O with structured result data and injected prompt/edit callbacks.

### Phase 3: Setup the CLI Layer (`mp13chat_tools_cli.py`) - Done
1.  Moved the implementation from `src/app/tools_cli_light.py` to `src/app/mp13chat_tools_cli.py`.
2.  Refactored it to act as a CLI router, instantiating the `SessionToolboxController` and implementing CLI-specific rendering and prompt callbacks.
3.  Kept command string parsing and command-to-controller dispatch in the CLI layer.
4.  Removed `tools_cli_light.py` after internal imports were updated.

### Phase 4: Consolidate External Handlers - Done
1.  Moved CLI-specific interactive handlers like the external tool prompt implementation and tools help renderer into `mp13chat_tools_cli.py`.
2.  Ensured `mp13chat.py` imports/delegates to the CLI handler and retains only the compatibility surface needed by existing load paths.

### Phase 5: Tests - Done
1.  Added focused unit tests for bridge/controller behavior and CLI parsing without requiring live terminal input.
2.  Added controller tests with fake prompt/edit callbacks and in-memory toolbox state.
3.  Preserved skip markers for tests that need external or spawned processes so normal Poetry test runs can avoid hangs when the environment is not configured.
4.  Ran focused Poetry tests, then the full Poetry test suite.

### Phase 6: Session Stack Helper Consolidation - Done
1.  Added session-level effective stack command replay in `EngineSession`, preserving `stack_id` for callers that need display context.
2.  Added `EngineSession.get_effective_tools_scope_entries(...)` for toolbox scope callers.
3.  Added `EngineSession.get_effective_adapter_entries(...)` for effective adapter/system-message display callers.
4.  Refactored redundant helpers in `session_cursor_toolbox.py` and `mp13chat.py` to use the session APIs.
5.  Confirmed `engine_session.py` does not import or depend on `context_cursor.py`.

## Expected Outcomes
-   **Architectural Integrity:** The engine layer (`mp13_engine`) remains completely unaware of app layer (`app`) constructs like cursors and sessions.
-   **Reusability:** Any new UI (like a web dashboard, desktop GUI, or automated testing rig) can directly use `SessionToolboxController` by providing a different set of I/O callbacks.
-   **Maintainability:** Tool management logic is cleanly separated from ANSI color codes and prompt loops.
-   **Cleanliness:** `mp13chat.py` will shed hundreds of lines of redundant or misplaced CLI logic.
