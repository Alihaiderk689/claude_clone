# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A local, free, Claude Code–style coding assistant CLI (`code-agent`) that runs entirely against a
local [Ollama](https://ollama.com) server and `qwen2.5-coder:3b` (default; `qwen2.5-coder:7b` also
works, see README) — no cloud API, no key. It has read-only project inspection tools (including
read-only Git inspection), approval-gated file-editing tools, one approval-gated command-execution
tool, four approval-gated Git tools (branch/stage/commit/push), and planning tools for multi-step
tasks (create_plan/update_plan/get_plan); the model can never write to disk, run a process, or
change Git state without going through an explicit user approval — a plan cannot skip that either,
it's just inert tracked data. As of Phase 7, the same agent is also reachable from a VS Code
extension via a local-only HTTP server (`code-agent serve`, `agent/server.py`) — see "The VS Code
integration layer" below; the terminal CLI itself is unchanged. As of Phase 8, failures at every
layer (tool, Ollama, subprocess, HTTP) are classified and recovered from instead of being allowed
to crash the agent or corrupt its state — see "Reliability and failure recovery (Phase 8)" below.
`git_push` (added after Phase 8, by explicit request) is the one deliberate exception to "no
remote operations" — see "Git push: narrow by design, not by accident" below for why it doesn't
weaken the no-generic-git-command guarantee. See README.md for the full user-facing walkthrough,
tool list, and phase-by-phase feature history.

## Commands

```bash
# Setup (from this directory)
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .            # installs the `code-agent` command

# Run the app (must be run from inside the target project directory —
# the cwd becomes the agent's project root)
cd /path/to/some/project && code-agent
# or, without the editable install:
python -m agent.cli

# Ollama must be running separately, with the model pulled:
ollama serve
ollama pull qwen2.5-coder:3b

# Tests (all mocked/temp-dir based — no Ollama server needed; subprocess is
# always mocked too, so no test ever actually spawns a process)
pytest                                  # full suite
pytest tests/test_loop.py               # one file
pytest tests/test_loop.py::TestApprovalFlow::test_approved_edit_is_applied  # one test
pytest -k "stale"                       # by keyword across files
pytest tests/test_server.py             # the local HTTP agent server (real sockets, mocked OllamaClient)

# Start the local agent server for the VS Code extension (same cwd rule as
# above -- but the server also accepts a workspace_root per request, see below)
cd /path/to/some/project && code-agent serve

# VS Code extension: separate TypeScript project, no vscode host needed for its tests
cd vscode-extension && npm install && npm run compile && npm test

# Debug logging (either the interactive CLI or serve mode)
code-agent --debug
code-agent serve --debug          # or: CODE_AGENT_DEBUG=1 code-agent serve

# Qwen 3B benchmark (real Ollama required; not part of the pytest suite)
python -m agent.benchmark --output benchmark_report.json
```

`OLLAMA_HOST` (default `http://localhost:11434`) and `OLLAMA_MODEL` (default `qwen2.5-coder:3b`)
are the only configuration, set as environment variables. `code-agent serve` reads the same two.

## Architecture

### Event-driven core, terminal-agnostic

`agent/loop.py`'s `run_agent_turn()` is a generator that drives one user turn (model call → optional
tool execution → repeat until a plain-text answer) and *yields structured event dicts* rather than
printing anything. `agent/cli.py` is the only module that imports `rich` and does terminal I/O; it
just consumes those events. Keep that split — new UIs (or a future non-interactive mode) should be
able to drive `run_agent_turn` without touching `cli.py`.

### The propose → confirm → apply pipeline (the core mechanism, used four times)

`edit_file`/`write_file` (`agent/tools/editing.py`), `run_command` (`agent/tools/terminal.py`),
`git_create_branch`/`git_stage`/`git_commit` (`agent/tools/git.py`), and `create_plan`
(`agent/tools/planning.py`) never touch disk, spawn a process, run Git, or adopt a plan themselves.
Each tool's `run()` only validates and builds a description of the pending action in memory —
`ProposedChange` (`agent/diff.py`), `ApprovedCommand` (`agent/command_policy.py`),
`ProposedGitOperation` (`agent/git_policy.py`), or a plain `Plan` (`agent/planner.py`) — set on
`ToolResult.pending_change` / `.pending_command` / `.pending_git_operation` / `.pending_plan`
respectively. When `run_agent_turn` sees any of those fields set, it does `approved = yield {"type":
"confirm"/"confirm_command"/"confirm_git_operation"/"confirm_plan", ...}` — a real pause using
Python's generator `send()` protocol. `cli.py` renders the diff/command/Git operation/plan, prompts,
and resumes the generator with `gen.send(approved)`. Only a `True` from that send leads to
`apply_change()` (atomic temp-file write + fsync + verify-by-read-back), `execute_command()`
(`subprocess.run([program, *args], shell=False, ...)`), `apply_git_operation()` (`git
branch`/`git add`/`git commit`, also never through a shell), or (for a plan) `run_agent_turn` itself
setting `task_state.plan = plan` directly, actually acting. There is no code path from the model
straight to disk, to a subprocess, to Git, or into `task_state` — a new mutating/executing/adopting
tool must follow this same propose/apply split, not act directly in its `run()`.

Because of this, `cli.py`'s event loop can't be a plain `for event in run_agent_turn(...)` — it uses
`gen.send(value)` manually so it can answer `confirm`/`confirm_command`/`confirm_git_operation`/
`confirm_plan` events (see `render_turn`). Note the asymmetries: file-edit approval supports
`y`/`n`/`a` (`a` = approve all remaining changes this turn); command and Git approval are strictly
`y`/`n` with no batch-approve; **plan approval is the one prompt in the whole app where a bare Enter
means yes** (`_ask_plan_approval` in `cli.py`) — deliberate, since a `Plan` object carries no path,
command, or Git ref, nothing capable of touching anything on its own. Don't "fix" that asymmetry;
it's intentional. Staging a file matching a sensitive pattern additionally swaps the normal "Stage
these files?" prompt for an explicit "WARNING: ... Are you sure ...?" one (still defaults to reject)
— see `_handle_git_confirm` in `cli.py`.

`command_policy.py`/`git_policy.py` (validation, allowlists, argument-shape rules, injection checks
— never execute anything) and `agent/tools/terminal.py`/`agent/tools/git.py` (execution only, given
an already-validated `ApprovedCommand`/`ProposedGitOperation` — never decide whether something is
allowed) are deliberately separate modules in both cases; don't blur that line when extending either.

**There is deliberately no generic "run a Git command" tool, and there must never be one for
destructive operations** (`reset`, `clean`, `checkout -- .`/`restore .`, `branch -D`, `rebase`,
`merge`, `push --force`). `git_policy.py` only knows how to validate a branch name, a list of
staging paths, a commit message, and (as of the narrow `git_push` tool) a remote name — there is no
function anywhere that accepts an arbitrary Git subcommand string. `git` is also explicitly in
`command_policy.py`'s `DENYLISTED_PROGRAMS`, so there's no back door through `run_command` either.

**Git push: narrow by design, not by accident.** `git_push` exists (added after Phase 8, by
explicit request) and it does NOT weaken the guarantee above, because of how narrowly it's scoped:
it only pushes whatever branch is *currently checked out* (`_current_branch()` — there's no
checkout tool, so this is never ambiguous or model-controllable) to a remote name that
`git_policy.py`'s `validate_remote_name` format-checks AND `agent/tools/git.py`'s `_git_push`
separately confirms is one `git remote` actually lists (the model cannot invent a remote or point
push at an arbitrary URL — there is no `git remote add` tool). The apply-side call in
`_apply_git_operation` is one hardcoded argv, `["push", "--set-upstream", op.remote,
op.branch_name]` — there is no parameter, flag, or code path anywhere that can append `--force`/
`-f`; `tests/test_tools_git.py::TestApplyGitPush::test_push_argv_never_contains_a_force_flag` spies
on the real argv passed to `_run_git` to lock this in, so a future refactor that accidentally adds
force-push capability would fail CI, not just review. If you're tempted to add a `force: bool` param
or a `refspec`/`branch` override to `GitPushArgs` — don't; that's exactly the shape of change that
would turn this from "narrow tool" back into "generic executor," which is the one thing this
codebase has held the line on since Phase 5.

### Tool system

`agent/tools/base.py` defines `Tool` (name + description + Pydantic args model + `run` callable) and
`ToolResult`. `ToolRegistry` (`agent/tools/registry.py`) turns registered tools into Ollama's
`tools` JSON-schema format (via each `Tool.args_model.model_json_schema()`) and dispatches calls by
name — arguments are always Pydantic-validated before a tool's `run()` ever sees them, and unknown
tool names come back as a normal failed `ToolResult` rather than raising. `build_default_registry()`
in `agent/tools/__init__.py` is the one place that wires up the current tool set; add new tools there.

### Path security is centralized, not per-tool

Every filesystem-touching tool resolves paths through `ProjectRoot.resolve()` (`agent/project.py`),
never doing its own path math. It rejects `..` traversal and absolute paths outside the root (careful:
naively joining an absolute path onto the root with `Path.__truediv__` silently discards the root —
`resolve()` handles this correctly, don't reimplement path joining elsewhere). `ProjectRoot` also owns
the ignored-directories list (`.git`, `node_modules`, `.venv`, etc.) and the sensitive-filename patterns
(`*.pem`, `*.key`, `id_rsa*`, `credentials*.json`, `secrets.*`, `*.sqlite3`) that `is_sensitive()` blocks
for both reading and writing. **`.env`/`.env.*` is deliberately *not* in `DEFAULT_SENSITIVE_PATTERNS`**
(by explicit request — the agent is allowed to create/edit `.env` directly, e.g. for "put the secrets
in .env" requests) — it's tracked separately via `ENV_FILE_PATTERNS`/`looks_like_env_file()`, checked
only by `git.py`'s `_git_stage` to keep the "you're about to stage a secrets-shaped file" warning intact
even though direct editing no longer needs it. If you touch either list, keep that split: don't merge
`.env` back into `is_sensitive()` (that would re-block editing) and don't remove `looks_like_env_file()`'s
check from `_git_stage` (that would silently drop the pre-commit warning — a real leaked-secret risk,
not just an inconvenience).
`command_policy.py` reuses the exact same `resolve()` for any non-flag `run_command` argument
(`_check_path_like_args_stay_in_project`) rather than reimplementing path containment checks, and
`git_policy.py`'s `validate_stage_paths()` does the same for `git_stage` — extend/reuse
`ProjectRoot.resolve()`, don't duplicate path-containment logic, if a new tool needs the same
guarantee.

### Stale-edit detection

`FileStateTracker` (`agent/tools/state.py`) is one shared instance per session, threaded through
`build_default_registry()` *and* `run_agent_turn(..., tracker=...)`. `read_file` records a content
hash on every read; `edit_file` checks it before proposing a change and refuses if the file drifted
on disk since it was last read (a path never read has nothing to compare against, so it's allowed).
`apply_change()` re-records the hash after a successful write so a follow-up edit in the same session
isn't wrongly flagged stale.

### Planning and task state: three cooperating, single-purpose modules

- `agent/planner.py` — plain data only (`Plan`, `PlanStep`, a `status` string, `render_lines()` for
  the `✓`/`→`/`○`/`✗` display). No behavior beyond that; don't add tool-calling or persistence logic
  here.
- `agent/task_state.py` — `TaskState`, the current task's short-term memory (goal, plan,
  files inspected/modified, recent commands, errors, Git actions). Its methods only mutate its own
  fields (e.g. `note_file_modified()` also removes that path from `files_inspected` — the
  cache-invalidation contract lives here) and produce `summarize()`, a bounded string. It never reads
  a `ToolResult` or a message list itself.
- `agent/context_manager.py` — the only module that bridges the two: `record_*` functions translate
  a tool's actual result into a `TaskState` update (called from `loop.py` right after each tool
  executes), `refresh_system_prompt()` regenerates (never appends to) the task-memory section of
  `messages[0]`, and `compact_messages()` mutates the raw `messages` list before each model call.

`TaskState` is deliberately *not* threaded through the tools themselves except `update_plan`/`get_plan`
(which read/write `task_state.plan` directly, since that's their whole job) — `read_file`,
`edit_file`, etc. stay exactly as they were pre-Phase-6; `context_manager.py` observes their results
from the outside instead of those tools needing to know about task memory at all. Keep that
separation if you add tracking for a new tool: extend `context_manager.py`'s `record_*` calls in
`loop.py`, don't reach into `TaskState` from inside `agent/tools/`.

`compact_messages()` pairs each `role: tool` message back to the real arguments that produced it by
walking `messages` and matching each `assistant` message's `tool_calls` to the `tool` messages that
follow, in order (`_iter_tool_messages_with_args`) — not by parsing any tool's output text. If you
add a new tool whose stale/duplicate results should also be compacted, extend that function's
`name == "..."` branches rather than adding per-tool string parsing.

`run_agent_turn()` calls `context_manager.compact_messages()` and `.refresh_system_prompt()` at the
**top of every iteration**, not just once per user turn — a single turn that calls several tools in a
row (a plan's whole point) can grow large before the model ever produces a final answer, so
compaction has to happen mid-turn, not just between turns.

### A real Ollama/qwen2.5-coder tool-calling quirk, handled in `loop.py`

Verified live against Ollama 0.32.8 with both `qwen2.5-coder:7b` and `:3b`: the model does not
reliably populate the API's structured `tool_calls` response field. Its own chat template documents
`<tool_call>{"name": ..., "arguments": {...}}</tool_call>` as its function-calling format, but in
practice the JSON often lands in plain assistant `content` instead — bare, tagged, or (observed
repeatedly) with free-form reasoning prose before it, with no tag at all. `_parse_fallback_tool_calls`
in `agent/loop.py` scans the full response text for that same model-documented JSON shape wherever it
appears and normalizes it into the same internal form a proper `tool_calls` reply would produce. This
is why each model response is fully buffered before display rather than streamed token-by-token —
there's no safe prefix check that rules out a tool call showing up later in the text, so nothing is
shown until the full response has been scanned. If you see this fallback firing unexpectedly, check
whether the model is echoing example JSON in prose rather than genuinely attempting a tool call before
assuming it's a bug.

### The model can narrate a fake tool result instead of calling the tool

Also observed live, with `qwen2.5-coder:3b` specifically: after a real approved `edit_file` call
succeeded, the model was asked to also run `pytest`, and instead of emitting a `run_command` tool
call it fabricated an entire fake `<tool_response>...Tests were run and all passed...</tool_response>`
block as plain text — no `confirm_command` event ever fired, no subprocess ever ran. The same thing
happened again during Phase 5 manual testing with a `git_stage` call: the model claimed success from
a malformed tool-call attempt (arguments shaped as a list instead of the expected object, so it
never validated) that never actually ran. This is a model honesty/reliability limitation, not a
security gap: nothing the model narrates in plain content can ever apply a change, run a command, or
touch Git by itself (see the pipeline above — only an actual `pending_change`/`pending_command`/
`pending_git_operation` on a real `ToolResult`, followed by real user approval, does that). But it
does mean **a chat transcript claiming success is not evidence of success** — if you're debugging
"why didn't my edit/command/Git operation happen," check for an actual `confirm`/`confirm_command`/
`confirm_git_operation` event in the transcript (or a `change_applied`/`command_result`/
`git_operation_applied` one), not just the model's prose, and independently re-check real state
(`git status`, the file on disk) rather than trusting either the model's or your own assumption. The
system prompt already instructs the model not to do this; a small model doesn't always comply.

### Rich markup escaping

Any dynamic text printed via `console.print()` (model output, tool args/errors, diff lines, paths)
must go through `rich.markup.escape()` or be printed with `markup=False`. Rich treats bare `[...]` as
a markup tag, so unescaped model output containing e.g. a Python list literal or slice silently loses
that text. This was a real bug found via live testing — every new print of dynamic content in `cli.py`
needs the same treatment.

### The VS Code integration layer (Phase 7): a transport, not a second implementation

`agent/server.py` is deliberately thin. It builds each workspace's session with the *exact same*
`_fresh_session_state()` helper `cli.py`'s own `/new` uses (imported from `cli.py`, not
reimplemented), and drives every turn with the *exact same* `run_agent_turn()` generator
`cli.py`'s `render_turn()` drives. If you're tempted to special-case behavior for the HTTP path,
stop — any actual agent behavior change belongs in `loop.py`/`tools/`, where both the CLI and the
server pick it up automatically. The server's own code should only ever be about: HTTP request/response
plumbing, per-workspace session bookkeeping, chunked NDJSON serialization, and auth.

**How an HTTP request can "pause" mid-generator.** `render_turn()` in `cli.py` pauses
`run_agent_turn()` on a `confirm*` event by blocking on `console.input()` in the same call stack — but
an HTTP server can't block a request handler on a *different, later* HTTP request. `agent/server.py`
solves this by keeping the `/chat` request's handler thread alive and blocked on a `queue.Queue`
(`Session.confirm_queue`) after writing the `confirm*` event to the response stream; a separate
`/chat/confirm` request (handled on its own thread, since `ThreadingHTTPServer` gives every request
its own thread) looks up the same `Session` and pushes the decision onto that queue, waking the first
thread back up to call `gen.send(decision)` and keep streaming on the *original* connection. Don't
"simplify" this into two independent request/response cycles — the generator genuinely has to stay
alive and paused in memory between the two HTTP calls, exactly as it stays alive on the Python call
stack between two `console.input()`-bounded moments in the CLI.

**Session isolation is by resolved absolute path, not by anything the client asserts.**
`SessionStore.get_or_create()` (`agent/server.py`) resolves `workspace_root` and uses that as the
dict key; two different paths always get two independent `Session`s with their own `ProjectRoot`,
`ToolRegistry`, `FileStateTracker`, `TaskState`, and `messages` list — nothing is shared or looked up
by anything the request claims about identity beyond the path itself. If you add server state that
should differ per workspace, put it on `Session`, not on a module-level global.

**`cancel_event` is additive everywhere it appears.** `OllamaClient.chat()`/`chat_stream()`/
`_chat_updates()` and `run_agent_turn()` all gained an optional `cancel_event: threading.Event = None`
parameter; every existing call site (the entire CLI) simply never passes one, so behavior there is
byte-for-byte unchanged. `run_agent_turn()` checks it at the *top of each iteration* (before the next
model call) and `_chat_updates()` checks it on every streamed line from Ollama, raising
`OllamaCancelledError` — caught in `run_agent_turn()` and turned into a `{"type": "cancelled"}` event,
a new terminal event type alongside `final`/`error`/`max_iterations`. If you add a new long-running
loop anywhere in the agent, thread `cancel_event` through it the same additive way rather than
inventing a second cancellation mechanism.

**Auth token file, not a hardcoded secret.** `run_server()`/`build_server()` generate a fresh
`secrets.token_hex(32)` on every start and write it to `~/.code-agent/server.json` (`chmod 600`,
via `os.open(..., 0o600)` then `os.chmod()` — not chmod'd after the fact from a wider default). Every
request except `GET /health` must present it as `Authorization: Bearer <token>`, checked with
`secrets.compare_digest` (not `==`, to avoid a timing side-channel). `BIND_HOST = "127.0.0.1"` is a
module-level constant with no parameter or env var that can override it — if you're adding
configuration to the server, keep that specific guarantee non-configurable.

**The VS Code extension (`vscode-extension/`) contains no agent logic.** `src/client.ts` only speaks
HTTP/NDJSON to the endpoints `agent/server.py` exposes (and deliberately has no `vscode` import, so
it's testable with plain `node:test`); `src/panel.ts` only renders webview state and forwards user
actions back over `postMessage`; `src/diffProvider.ts` only visualizes a change's `old_content`/
`new_content` that the server already sent, via `vscode.diff` — it never writes a file. If a feature
request means the extension would need to decide something about editing, running a command, or
approving anything, that decision belongs in the Python agent, with the extension only sending the
user's answer to `/chat/confirm` — same as how `cli.py` never decides `apply_change()`'s outcome
itself, only whether to call it.

### Reliability and failure recovery (Phase 8)

**The dispatch boundary, not each tool, is where "never crash" is enforced.** `Tool.execute()`
(`agent/tools/base.py`) is the one place every tool call passes through, and it already caught
`ToolError` and any other exception before Phase 8 — that safety net didn't need building, only
extending with classification (see below). The thing Phase 8 actually found and fixed was a
*different* code path that had no such net at all: `apply_change()`, `execute_command()`, and
`apply_git_operation()` are called **directly from `loop.py`**, after user approval, bypassing
`Tool.execute()` entirely (they're not tool `run()` calls, they're the actual state-changing action).
The first two were already internally safe (`apply_change` catches `OSError` around its write/verify;
`execute_command` never lets a subprocess exception escape). `apply_git_operation` was not — a
`git` binary vanishing mid-session or a hung `git` process would have raised straight out of the
agent loop's generator. Fixed by giving `apply_git_operation` its own `try/except ToolError` wrapper,
exactly mirroring the shape `Tool.execute()` already had. **If you add a fourth "apply" function called
directly from `loop.py` after approval, it needs this same self-contained guard — it will never get
`Tool.execute()`'s protection for free.**

**Error classification piggybacks on the existing `ToolError` hierarchy, not a parallel one.**
`ToolError` itself now carries `error_type: str = "ToolExecutionError"` and `recoverable: bool = True`
class attributes; a small set of subclasses in `agent/tools/base.py` (`ValidationFailedError`,
`NotFoundError`, `PermissionDeniedError`, `StaleStateError`, `ToolTimeoutError`,
`ExternalToolUnavailableError`) override them. `Tool.execute()` copies whichever values the raised
exception carries onto the resulting `ToolResult`'s new `error_type`/`recoverable` fields (both
`Optional`, default `None` — every pre-Phase-8 hand-built `ToolResult(ok=False, output=...)` call site
that was never touched still works identically, just "unclassified" rather than wrong). Raising a bare
`ToolError` anywhere still works and still gets a sane generic classification — you are never required
to pick a subclass, only encouraged to when the distinction is actually actionable for the model or
for logs.

**`git.py` has two repo-check functions on purpose, not by accident: `is_git_repository()`
(never raises, collapses "not a repo" and "couldn't check" into one `False`) and
`_check_git_repository()` (lets a `ToolError` propagate).** The read-only tools' `_require_git_repo`
and `apply_git_operation` both need `_check_git_repository()` — they have real classification to offer
a caller (or, for `apply_git_operation`, their own `except ToolError` wrapper) if git itself is
unavailable/timed out, and collapsing that into a generic "not a Git repository" message would
actively mislead the model into thinking `git init` is the fix. `is_git_repository()` still exists,
unchanged in contract, for callers (and its own pre-Phase-8 tests) that only want a plain boolean with
no more specific handling to offer. Don't merge these back into one function.

**Repetition/failed-approach detection lives in `run_agent_turn`'s tool-call loop, not in a
separate module.** A `(name, json.dumps(args, sort_keys=True))` signature is tracked across
*iterations* within one turn (not reset per model round-trip) via `last_call_signature`/
`consecutive_call_count`, declared once before the `for _ in range(max_iterations)` loop. On the
`MAX_CONSECUTIVE_IDENTICAL_CALLS`-th (3rd) consecutive identical call, the loop skips
`registry.execute()` entirely — no real (re-)execution, no repeated approval prompt — and injects a
synthetic tool result telling the model to change approach, yielding `{"type":
"repetition_detected"}`. This one mechanism intentionally covers both a call that keeps "succeeding"
with no progress (e.g. re-reading the same file) and one that keeps failing the same way (e.g.
re-proposing a rejected edit) — they're the same signature-repeats-3-times pattern from this code's
point of view. A different, non-identical call resets the streak immediately.

**Ollama retries are a `while True` around the model call inside the per-iteration loop, not a
decorator or a separate retry module.** Only `OllamaConnectionError`/`OllamaTimeoutError` retry (up to
`MAX_OLLAMA_RETRIES`, backoff `OLLAMA_RETRY_BACKOFF_SECONDS`); `OllamaModelNotFoundError`/
`OllamaAPIError` fail immediately since retrying an identical request can't help. The backoff sleep
(`_interruptible_sleep`) wakes every 50ms to check `cancel_event`, so Stop Task stays responsive even
mid-backoff, not just mid-generation. Tests that exercise this **must** monkeypatch
`OLLAMA_RETRY_BACKOFF_SECONDS` to `(0, 0)` or they will actually sleep for real seconds.

**`execute_command` uses `Popen` + polled `communicate(timeout=...)` calls, not one blocking
`subprocess.run(..., timeout=...)`, specifically so it can react to `cancel_event` mid-command, not
just to its own timeout.** It also passes `start_new_session=True` so `_kill_process_group()` can
`os.killpg()` the whole process group (SIGTERM, then SIGKILL after `PROCESS_TERMINATE_GRACE_SECONDS`)
instead of `proc.kill()`-ing only the direct child — this matters for anything that spawns its own
worker subprocesses (a parallelized test runner, for instance). If you touch this function, keep
running `TestRealSubprocessTermination` in `tests/test_tools_terminal.py` — it runs a real `sleep 30`
and asserts the OS process is actually dead afterward, not just that Python stopped waiting for it.

**Logging (`agent/logging_config.py`) is opt-in and silent by default on purpose.** `get_logger(name)`
can be called and logged to at import time by any module (`loop.py`, `server.py` both do) with zero
effect on any existing caller — nothing is configured (no handler, WARNING level, matching stdlib's
own default) until an entry point (`cli.py`'s `main()` / `_run_serve_command`) explicitly calls
`configure_logging(debug=...)`. Every message passes through a redaction filter first
(`_RedactingFormatter`) that scrubs `Bearer <token>`-shaped and `key=value`-shaped secrets — tuned
with an 8-character minimum specifically so it doesn't mangle plain sentences that happen to contain
the word "token" (a real false positive caught during Phase 8 manual testing: "Missing or invalid
Authorization bearer token." was getting partially redacted before the length threshold was added).
If you add a new secret-shaped pattern, keep that same defense-in-depth framing — the actual guarantee
is still "this codebase doesn't log raw request bodies/env dumps/file contents," not the regex.
