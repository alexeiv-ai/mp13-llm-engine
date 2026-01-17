# App Layers and Architecture Orientation

This document is an orientation guide for advanced users, applied researchers, and developers. It explains how MP13 separates the **core engine** from the **app/context runtime**, and how higher-level workflows (branching, replay, tool scoping) are modeled.

- If you want to *use* MP13-engine: start with **README.md**, then **INSTALL.md** and **CONFIG.md**.
- If you want to *extend* MP13-engine (UI, orchestration, MCP server, new workflows): this doc is a good starting point.

---

## The split: engine vs app layers

### Core engine (`mp13_engine/`)
Production-oriented:
- Runs inference and LoRA training
- Owns tool-call parsing semantics shared between testing and deployment
- Provides async/concurrency/cancellation down to HF `generate`
- Exposes metrics (per-request and rolling aggregates)

### App/context layers (MIT)
Experimentation-oriented today:
- Implement a **potentially dynamic context-engineering paradigm**
- Provide a full system: session trees, cursors, branching, replay, CLI orchestration
- API/design may evolve based on community feedback

---

# App Layers

This document summarizes the core capabilities of the session foundation and the
application-facing API layer. `context_cursor.py` sits on top of the core
`engine_session.py` model and proxies formatting and turn mutations to the session
while owning app-level runtime state.

## Foundation Layer: EngineSession (Core Session Model)

**Purpose (primary ownership)**
- Owns the in-memory session tree for one or more conversations (`ChatSession`).
- Defines and enforces the canonical turn/command structure and tree consistency.
- Provides concurrency safety for structural state via a reader/writer lock.

**Concurrency and consistency guarantees**
- The session lock protects structural collections and links: conversation list,
  turn-tree links, per-turn `cmd`, and `commands_history`.
- Turn payloads (`Turn.data`) are not locked; callers own their safety.
- Root turns are reserved anchors and remain structural (never reused as chat).

**Turn content model and formatting consistency**
- Turn content is normalized as ordered payloads (`user`, `tool_results`, `assistant`)
  with explicit role handling and tool-call block support.
- Tool-call blocks and tool results are rehydrated to preserve API compatibility.
- Message materialization filters placeholders and structural nodes to keep LLM
  prompts consistent and deterministic across branches.
- Continuation and truncation are stitched into a single assistant response to
  preserve linear prompt semantics.

**Serialization and persistence**
- Full session serialization/deserialization of turns, commands, and tool payloads.
- Stable gen_id and command_id assignment for replay and external indexing.

**Core data structures**
- `Turn`: a node in the conversation tree with user/tool/assistant payloads, flags
  (archived, main thread, truncated, canceled, continuation), metrics, and children.
- `Command`: a structured record of non-chat state changes (param/state/adapters/tools),
  including API call metadata for replay or auditing.
- `ChatSession`: a single conversation tree with a root placeholder, engine config,
  parser profile, toolbox, inference defaults, and title/metadata.
- `InferenceParams`: per-session defaults for inference requests with serialization
  that omits empty values.

**Conversation and branch management**
- Multiple conversations per engine session with explicit root anchors.
- Structural nodes for `FORK` and `BATCH`, plus placeholder/hold/reserved anchors.
- Add user/assistant turns, tool results, and continuation turns without reloading.
- Batch and fork helpers to split, close, and promote branches to the main thread.
- Try-out support (create, close, promote) with placeholder anchors for experiments.

**Command kinds (session-owned history of state changes)**
- `COMMAND`: general purpose command record (used for structured side effects).
- `LOG`: client-side notes that do not call the engine API.
- `PARAM_CHANGE`: explicit parameter changes (system message, generation config, etc).
- `STATE_CHANGE`: implicit behavior changes (flags, tools activation, adapters toggles).
- `ADAPTERS_STATE`: reconstructed adapter state snapshot for replay or migration.
- `TURN`: a turn wrapper used for annotating conversational exchanges as commands.

**Other capabilities**
- System message stacking/segmentation and effective system resolution per turn.
- Adapter state tracking and per-turn adapter commands.
- Tool scope activation, tools access resolution, and tool-call serialization.
- Parser profile integration for tool-call rehydration and message reconstruction.
- Debug-friendly formatting with optional content previews and trace info.
- Turn-level metrics storage and incremental updates.
- Command history snapshotting and pruning.
- API call recording for param/state changes and inference requests.

## App Layer API: context_cursor (ChatContext and Cursor Runtime)

**Purpose (sits on top of EngineSession)**
- Owns app-level runtime state and cursor management while delegating structural
  mutations and message formatting to `EngineSession`.
- Adds cursor-based navigation and scoped state to make branching, retries, and
  tool/adapters configuration safe for apps.

**ChatContext (app-level session manager)**
- Owns cursor registry, active cursor resolution, and scope-aware cursor lookups.
- Resolves or creates a `ChatSession`, then proxies turn mutations to `EngineSession`.
- Maintains global defaults (streaming, cache, tool parsing, etc) for new cursors.
- Persists tools scope on the root turn to keep app-wide tool visibility consistent.
- Try-out anchor registry with scope-based isolation and controlled promotion.
- Auto-iteration queue for orchestration/automation cycles.
- Concurrency safety: context-level lock protects cursor registry, anchors, and
  scope-tracked collections (it does not lock turn payloads).

**ChatCursor (branch-level runtime)**
- Owns the moving head pointer for a branch and clones/snapshots for safe traversal.
- Manages stacked operations with dynamic resolution per active branch.
- Proxies turn mutations and formatting requests to `EngineSession` while recording
  local overrides and app-level defaults.
- Builds inference requests by combining global defaults + cursor overrides +
  per-branch resolved state.
- Provides branch navigation helpers (parent/child/siblings/leaf) and branch closing.
- Command capture for app-level intent (log/param/state/adapters/tools) stored via session.

**Stacked context commands**
- System message stack
  - App management: push/replace system message segments on a cursor.
  - Result: session computes an effective system message by walking the active path.
- Adapter stack
  - App management: add adapter commands and record adapter state on turns.
  - Result: effective adapters are resolved per branch for consistent tool/inference routing.
- Tools scope stack
  - App management: apply tools-scope commands that mutate advertise/silent/disable sets.
  - Result: tools visibility/permission view is resolved per branch and request.
- Parameter/flag overrides
  - App management: cursor overrides for stream/cache/return_prompt/etc.
  - Result: request payloads combine context defaults + cursor overrides + per-turn changes.
- Fork/batch stack frames
  - App management: open/close fork and batch contexts on a cursor.
  - Result: nested branching stays isolated; state resolution follows the active branch.

**Dynamic resolution across branches (dynamic prompt context)**
- Effective system message, adapters, tools scopes, and flags are computed from
  the active path, so the same cursor on different branches yields different state.
- Try-out anchors and fork/batch stacks prevent cross-branch leakage of overrides.

**ChatForks (batch result management)**
- Tracks per-prompt cursors created by batch operations.
- Updates and resolves the active cursor for a batch.
- Supports completion detection and main-placeholder promotion.

**ChatContextScope (app isolation boundary)**
- Owns a per-scope active cursor and allows per-scope cursor overrides.
- Prevents cross-scope cursor reuse and cross-branch try-out leaks.
- Enables parallel app components to share a session safely without state collisions.
- Scope-local bag storage for lightweight app state.

## Tools Layer: Toolbox (mp13_toolbox.py)

**Purpose**
- Central registry and executor for intrinsic and user-defined tools.
- Builds per-request tool visibility/permission views from global mode and scopes.

**Tool registry and modes**
- `Toolbox`: the central registry for tool definitions and callables, with a global
  mode (`advertised`, `silent`, `disabled`) and active/hidden sets for tool control.
- Supports intrinsic tools, user-defined tools, and guide tools.
- Active/hidden sets for intrinsic and user tools, with intrinsic override support.

**Core types**
- `ToolsScope`: scoped mutation of tool visibility (advertised/silent/disabled)
  with named allow/hidden/disabled sets and optional labels.
- `ToolsView`: materialized, per-request permissions snapshot derived from scopes
  and restricted to the permissions set on the owning toolbox instance.
- `ToolsAccess`: lightweight wrapper that memoizes a `ToolsView` for reuse.
- `ToolBoxRef`: toolbox reference that persists a tracked scope snapshot.

**View construction**
- `build_view(...)` resolves the effective mode and per-tool status from a stack
  of `ToolsScope` objects, including wildcard targeting (`*`).
- Produces advertised vs hidden-but-allowed vs disabled tool sets for a request.

**Execution model**
- Validates tool availability against the current `ToolsView` or global mode.
- Executes intrinsic or user callables (sync or async) and serializes results.
- Handles malformed tool arguments with recovery paths and error tagging.
- Supports guide tools and auto tool-call execution across batches.

**Serialization and relinking**
- Toolbox state serializes to/from dicts for session persistence.
- User tool callables are re-linked on load via provided search scopes or handlers.

## Chat Module: (CLI Orchestration)

**Primary APIs used**
- Engine API via `call_api(...)` (wrapper around `handle_call_tool`) for inference,
  cancellation, and engine-level operations.
- Session/context API via `ChatContext`/`ChatCursor` (and `EngineSession` formatting)
  for in-memory state, turn creation, replay, and branch-aware formatting.

**Main usage scenarios**
- Interactive CLI chat loop with slash commands and live user prompts.
- Branching workflows (forks, batches, try-outs) to explore alternatives safely.
- Session load/save and replay to reproduce or migrate prior runs.
- Load a session under a different configuration and continue from a new model
  by selecting config paths or overrides at startup.
- Wrapper apps can host `mp13chat` and register external tool callables into the shared toolbox before the chat loop starts.
- Tool-enabled inference with automatic tool execution rounds.

**Advanced controls**
- CLI flags for configuration selection, quantization overrides, torch.compile toggle,
  and initialization dumps.
- Replay supports running a session under a different configuration than the source,
  including manual model selection or command-line overrides.
- Runtime parameter overrides (streaming, cache, generation templates, tool parsing).
- Tools visibility and mode controls (advertised/silent/disabled) plus scope stacks.
- Auto-retry workflows for tools and truncated responses, with retry limits.
- Cancellation support via Ctrl+C and the `cancel-request` API.

**Debugging and inspection**
- Console and file logging with selectable verbosity and colored output.
- Stored exception tracebacks surfaced via `/f ex` in the CLI.
- `/fp` prompt formatter for detached prompt previews (with continue/root variants).
- Multiple history views: compact summaries, full tree views, and command history
  via `/s` history helpers (including per-conversation views).
- Tools scope summaries and try-out anchor summaries for current context.
- Replay debug paths with optional verbose tracing for batch and branch replay.
