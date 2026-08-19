# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A local, free, Claude Code–style coding assistant CLI (`code-agent`) that runs entirely against a
local [Ollama](https://ollama.com) server and `qwen2.5-coder:3b` (default; `qwen2.5-coder:7b` also
works, see README) — no cloud API, no key. It has read-only project inspection tools (including
read-only Git inspection), approval-gated file-mutation tools (edit_file/write_file/delete_file/
rename_file), one approval-gated command-execution tool, four approval-gated Git tools
(branch/stage/commit/push), and planning tools for multi-step tasks (create_plan/update_plan/
get_plan); the model can never write to disk, delete or rename a file, run a process, or change Git
state without going through an explicit user approval — a plan cannot skip that either, it's just
inert tracked data. As of Phase 7, the same agent is also reachable from a VS Code extension via a
local-only HTTP server (`code-agent serve`, `agent/server.py`) — see "The VS Code integration
layer" below; the terminal CLI itself is unchanged. As of Phase 8, failures at every layer (tool,
Ollama, subprocess, HTTP) are classified and recovered from instead of being allowed to crash the
agent or corrupt its state — see "Reliability and failure recovery (Phase 8)" below. `git_push`
(added after Phase 8, by explicit request) is the one deliberate exception to "no remote
operations" — see "Git push: narrow by design, not by accident" below for why it doesn't weaken the
no-generic-git-command guarantee. Both the CLI and the VS Code extension also support a Manual/Auto
approval-mode toggle (added by explicit request, see "Manual/Auto approval mode" below) — Auto only
ever auto-approves file edits/deletes/renames and plan approvals, never commands or Git. Phase 9
closed real (verified, not assumed) context/performance gaps on top of Phases 1–8's already
substantial context management — see "Phase 9: context, performance & Qwen 3B optimization" below
for exactly what was already there versus what's genuinely new. Phase 10 added `delete_file`/
`rename_file` as first-class approval-gated tools (replacing "the agent literally cannot rename or
remove a file" with a validated, structured alternative to routing through `run_command`'s
`rm`/`mv`, which stay denylisted), a narration-vs-tool-call detection/correction mechanism in the
agent loop, a cache-hit response redesign, and Ollama generation-parameter tuning aimed at more
reliable tool-calling on the 3B model — see "Phase 10: autonomous filesystem operations and
tool-calling reliability" below. See README.md for the full user-facing walkthrough, tool list, and
phase-by-phase feature history.

## Operating principles for Claude Code sessions on this repo

This project is developed across two machines with different Ollama models (a RAM-limited laptop
running `qwen2.5-coder:3b`, and a second machine that can also run `qwen2.5-coder:7b` for later
validation). Because of that, code changes here must never depend on how capable the *model
generating them* happens to be — prefer deterministic tools, explicit workflows, and small
readable functions over anything that relies on a model "remembering" or "figuring out" state.
This applies to both layers: the qwen model this agent drives at runtime (see the
`SYSTEM_PROMPT_TEMPLATE` sections below for that layer) and Claude Code sessions editing this
codebase itself.

When working in this repo, actually do the work rather than describing it:
- Inspect before editing — read a file and find the real, existing text before calling an edit
  tool; never invent a placeholder path and call a tool with it unverified.
- If a tool call fails, that's information to act on, not a cue to retry the same call — change
  strategy (re-inspect, pick a different tool, ask only if genuinely blocked on something only the
  user can decide).
- Make the actual file changes, run the real test suite (`pytest`, `vscode-extension && npm test`),
  and fix what testing turns up — don't report an error and stop, and don't claim something was
  changed or verified without having actually run the check.
- Preserve the existing architecture and naming conventions (see "Architecture" below) rather than
  rewriting adjacent code the request didn't ask about.
- Never claim a test passed, a file changed, or a bug is fixed without having actually run the
  check in this session — a plausible-sounding result is not a verified one.

Code written here is read next by whichever model opens the repo next, possibly the 7B one on the
other machine — so favor clear function names, small single-purpose functions, predictable control
flow, and no hidden state over anything clever, even if the clever version is shorter. Add a
comment only where the *why* isn't obvious from the code itself (a workaround, a non-obvious
constraint); don't restate what a well-named function already says.

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
python -m agent.benchmark --compare before.json after.json  # diff two saved reports, no Ollama needed
```

`OLLAMA_HOST` (default `http://localhost:11434`), `OLLAMA_MODEL` (default `qwen2.5-coder:3b`),
`OLLAMA_TIMEOUT` (default `120` seconds, parsed by `ollama_client.py`'s `timeout_from_env()` --
falls back to the default rather than raising on a missing/non-numeric/non-positive value), and
`OLLAMA_KEEP_ALIVE` (default `30m`, parsed by `keep_alive_from_env()` following the identical
tolerant-fallback pattern -- passed through as-is to Ollama's `keep_alive` field on every
`/api/chat`/`/api/generate` call) are the only configuration, set as environment variables.
`code-agent serve` reads the same four.

**Cold model load is a real, previously-documented failure mode, not just a perf nit.** Ollama's own
server default `keep_alive` is `5m` -- short enough that a normal pause between agent turns (reading
a diff, deciding what to ask next) lets the model fall out of memory, so the next `/api/chat` call
pays a multi-second-to-tens-of-seconds reload. Verified live (`ollama ps` / `load_duration` in the
response) against `qwen2.5-coder:3b`: a genuinely cold load took ~6.4s of a ~7.5s total response
time, vs. ~0.1s `load_duration` when already resident. On slower/more memory-constrained hardware or
the 7B model this can exceed `OLLAMA_TIMEOUT`, which is what most "Connection to Ollama failed...
Retrying" messages actually are -- not a real connectivity problem. Two fixes address this instead of
just papering over it with a bigger timeout: `OllamaClient` now sends `keep_alive=OLLAMA_KEEP_ALIVE`
(`30m` default, up from Ollama's own `5m`) on every chat request, so the model stays resident for the
length of a normal session; and `OllamaClient.warm_up()` (a promptless `/api/generate` call, which
Ollama loads the model for but never generates tokens against) is called once at CLI startup, behind
a "Loading model into memory..." status spinner, right after `check_connection()` succeeds -- so the
cold-load cost is paid once, up front, with honest messaging, instead of silently during the user's
first real message where it looks like a failure. `warm_up()` failures are caught and shown as a
yellow warning, not a hard exit -- pre-loading is a latency optimization, not a correctness
requirement, and the existing per-turn retry loop in `loop.py` still covers a cold load that happens
anyway.

## Architecture

### Event-driven core, terminal-agnostic

`agent/loop.py`'s `run_agent_turn()` is a generator that drives one user turn (model call → optional
tool execution → repeat until a plain-text answer) and *yields structured event dicts* rather than
printing anything. `agent/cli.py` is the only module that imports `rich` and does terminal I/O; it
just consumes those events. Keep that split — new UIs (or a future non-interactive mode) should be
able to drive `run_agent_turn` without touching `cli.py`.

### The propose → confirm → apply pipeline (the core mechanism, used five times)

`edit_file`/`write_file`/`delete_file`/`rename_file` (`agent/tools/editing.py`,
`agent/tools/file_ops.py`), `run_command` (`agent/tools/terminal.py`), `git_create_branch`/
`git_stage`/`git_commit` (`agent/tools/git.py`), and `create_plan` (`agent/tools/planning.py`) never
touch disk, spawn a process, run Git, or adopt a plan themselves. Each tool's `run()` only validates
and builds a description of the pending action in memory — `ProposedChange` (`agent/diff.py`),
`ProposedFileOp` (`agent/file_ops.py` — delete/rename only; no textual diff to show, so it's a
separate dataclass rather than being folded into `ProposedChange`), `ApprovedCommand`
(`agent/command_policy.py`), `ProposedGitOperation` (`agent/git_policy.py`), or a plain `Plan`
(`agent/planner.py`) — set on `ToolResult.pending_change` / `.pending_file_op` / `.pending_command` /
`.pending_git_operation` / `.pending_plan` respectively. When `run_agent_turn` sees any of those
fields set, it does `approved = yield {"type":
"confirm"/"confirm_file_op"/"confirm_command"/"confirm_git_operation"/"confirm_plan", ...}` — a real
pause using Python's generator `send()` protocol. `cli.py` renders the diff/path-or-rename/command/
Git operation/plan, prompts, and resumes the generator with `gen.send(approved)`. Only a `True` from
that send leads to `apply_change()` (atomic temp-file write + fsync + verify-by-read-back),
`apply_file_op()` (`os.remove`/`os.rename`, each re-checking the target still exists / destination
still doesn't right before acting — same stale-proposal defense as `apply_git_operation`'s
re-checked staged-files/branch), `execute_command()` (`subprocess.run([program, *args], shell=False,
...)`), `apply_git_operation()` (`git branch`/`git add`/`git commit`, also never through a shell), or
(for a plan) `run_agent_turn` itself setting `task_state.plan = plan` directly, actually acting.
There is no code path from the model straight to disk, to a subprocess, to Git, or into `task_state`
— a new mutating/executing/adopting tool must follow this same propose/apply split, not act directly
in its `run()`.

Because of this, `cli.py`'s event loop can't be a plain `for event in run_agent_turn(...)` — it uses
`gen.send(value)` manually so it can answer `confirm`/`confirm_file_op`/`confirm_command`/
`confirm_git_operation`/`confirm_plan` events (see `render_turn`). Note the asymmetries: file-edit
and file-op (delete/rename) approval share one `y`/`n`/`a` prompt and one `turn_state["approve_all"]`
flag (`a` = approve all remaining edits/deletes/renames this turn — the same risk class, see
`_handle_file_op_confirm` in `cli.py`); command and Git approval are strictly `y`/`n` with no
batch-approve; **plan approval is the one prompt in the whole app where a bare Enter means yes**
(`_ask_plan_approval` in `cli.py`) — deliberate, since a `Plan` object carries no path, command, or
Git ref, nothing capable of touching anything on its own. Don't "fix" that asymmetry; it's
intentional. Staging a file matching a sensitive pattern additionally swaps the normal "Stage these
files?" prompt for an explicit "WARNING: ... Are you sure ...?" one (still defaults to reject) — see
`_handle_git_confirm` in `cli.py`.

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

### Manual/Auto approval mode: scoped to edits (including delete/rename) and plans, never commands or Git

Both the CLI (`/auto`, `/manual`) and the VS Code extension (the Manual/Auto toggle in the status
bar) offer a persistent mode that changes how `confirm`/`confirm_file_op`/`confirm_plan` events are
resolved — nothing else. **Auto mode auto-approves file edits, deletes, renames, and plan approvals
only. `run_command` and every Git operation (including `git_push`) always require individual
approval, regardless of mode, with no way to disable that.** `delete_file`/`rename_file` were added
to this same scope deliberately, not as an oversight: a delete/rename is exactly the same risk class
as an edit — a filesystem mutation, reversible via Git if the project is one, no command execution,
no Git state change — so it belongs in the same auto-approval bucket rather than either forcing a
manual approval Auto mode was supposed to remove, or (the wrong direction) folding it into the
command/Git bucket that must never be auto-approved. This mirrors the pre-existing `approve_all`/`a`
"approve all" mechanism in `cli.py`'s `_handle_confirm`/`_handle_file_op_confirm` (typing `a` at an
edit OR a delete/rename prompt approves all remaining edits/deletes/renames this turn, sharing one
`turn_state["approve_all"]` flag), which was likewise deliberately never extended to commands or
Git — Auto mode is that same scope decision made persistent and explicit instead of a new risk
categorization.

The actual mechanism lives entirely in `run_agent_turn()` (`agent/loop.py`), which takes an
additive `auto_approve_edits: bool = False` parameter (default is a no-op for every existing
caller/test). In the `confirm`, `confirm_file_op`, and `confirm_plan` blocks only, when
`auto_approve_edits` is set, `approved` is resolved to `True` **before** the yield rather than
depending on whatever the caller sends back:

```python
if auto_approve_edits:
    approved = True
    yield {"type": "confirm", "name": name, "change": change, "auto_approved": True}
else:
    approved = yield {"type": "confirm", "name": name, "change": change}
```

The event is still emitted either way (so any caller can render what happened), but the approval
decision no longer depends on caller cooperation — even a caller unaware of `auto_approved` can, at
worst, show a redundant prompt, never accidentally skip a real approval or block forever. The
`confirm_command` and `confirm_git_operation` blocks are untouched — no parameter reaches them, so
there is no code path by which Auto mode can affect a command or a Git operation.

`agent/server.py`'s `Session` dataclass carries the mode (`mode: str = "manual"`), set via
`POST /task/mode` and reported back through `GET /task/status`; `SessionStore.reset()` deliberately
does not touch it, so the mode persists across `/task/new`/New Task (it's a UI preference, not task
data) but always starts `"manual"` the first time a session is created — safety-by-default on every
fresh `code-agent serve` start or newly opened workspace. `cli.py` keeps the equivalent as a plain
local variable in `main()`'s loop, toggled by `/auto`/`/manual`, with the same survive-`/new`,
reset-on-restart behavior. If you add a new confirm-shaped event type in the future, default it to
requiring approval and only wire it into `auto_approve_edits` after deliberately deciding it belongs
in the same risk class as a file edit — not by default inclusion.

### Tool system

`agent/tools/base.py` defines `Tool` (name + description + Pydantic args model + `run` callable) and
`ToolResult`. `ToolRegistry` (`agent/tools/registry.py`) turns registered tools into Ollama's
`tools` JSON-schema format (via each `Tool.args_model.model_json_schema()`) and dispatches calls by
name — arguments are always Pydantic-validated before a tool's `run()` ever sees them, and unknown
tool names come back as a normal failed `ToolResult` rather than raising. `build_default_registry()`
in `agent/tools/__init__.py` is the one place that wires up the current tool set; add new tools there.

### Path security is centralized, not per-tool

Every filesystem-touching tool resolves paths through `ProjectRoot.resolve()` (`agent/project.py`),
never doing its own path math. **It also rejects an unfilled placeholder path (anything containing
`<`/`>`, e.g. `read_file(path="<file_path>")`)** — observed live from `qwen2.5-coder:3b`, which
retried that exact literal several times before the pre-existing repetition guard even kicked in.
Angle brackets never appear in a real path on any platform this runs on, so this is a safe, cheap
check to add here once rather than teaching every tool to recognize a template token, and the
`PathSecurityError` message directs the model at `list_files`/`search_files` instead of a bare "not
found" that invites retrying the same placeholder.

**The same failure mode showed up one argument later, and needed a separate, narrower guard.**
Once the path guard above was in place, live testing surfaced the model clearing that hurdle
(`list_files` → `read_file` → real content in hand) and then calling `edit_file` with
`old_text='<existing text to replace>'`/`new_text='<new content here>'` instead of copying the real
text it just read. `agent/tools/editing.py`'s `_looks_like_placeholder_token()` catches this, but
deliberately *cannot* reuse `ProjectRoot`'s "contains `<`/`>`" rule — `old_text`/`new_text` are
arbitrary source code, where `<`/`>` legitimately appear (comparisons, generics, HTML), unlike a
path. Its regex (`^<[^<>\n]{1,200}>$`) only matches when the *entire* trimmed argument is nothing
but one bracketed phrase, which real code is never shaped like even for a single short line —
`tests/test_tools_editing.py::TestEditFilePropose::
test_real_old_text_containing_angle_brackets_is_still_allowed` locks in that a genuine `x < 10 and
x > 0` edit still works. The resulting `ToolError` names which argument was the placeholder and
points at the specific `read_file(...)` call to copy from, rather than the generic "target text not
found" a real-but-wrong `old_text` gets — that generic message was tried first, live, and wasn't
specific enough to redirect the model away from repeating the same placeholder shape.

It rejects `..` traversal and absolute paths
outside the root (careful:
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

### Proposed Python content is syntax-checked before it's ever shown as a diff

Everything above catches a model sending the *wrong kind of argument* (a placeholder path, a
placeholder block of content). It does not catch a model sending a *real, well-intentioned* new/
new_text value that simply isn't valid Python — live-observed in the exact same session as the
placeholder failures above: `qwen2.5-coder:3b` generated a `bubble_sort()` body where every nested
line (the function body, the `for` loop, the nested `for` loop, the `if` statement) used the
identical single-space indent instead of increasing per block — a genuine `IndentationError`, not a
style choice. Nothing before this checked that the resulting file content would actually parse, so
that content would have been shown to the user as an approvable diff and, on approval, written to
disk as broken Python — "the tool call succeeded" and "the file is valid Python" are different
claims, and only the second one actually matters to the user.

`agent/tools/editing.py`'s `_validate_python_syntax(path, content)` closes this: both `_edit_file`
(against the post-edit `new_normalized` content) and `_write_file` (against `args.content`) run it
right before building the `ProposedChange`, so a syntax error is caught *before* proposal, not
after approval. It's a thin wrapper around stdlib `ast.parse()` — no new dependency, consistent
with this project's "don't add a dependency without reason" rule — which also catches
`IndentationError`/`TabError` for free since both subclass `SyntaxError`. The resulting
`ValidationFailedError` reports the real line/column and message from the parser and names the
likely cause (inconsistent indentation) rather than a generic "invalid content," giving the model
something concrete to fix rather than just a reason to guess again.

**Deliberately scoped to `.py` files only** (`path.endswith(".py")`) — this agent's tools are
otherwise language-agnostic, and there's no general multi-language syntax checker to extend this
to; `tests/test_tools_editing.py`'s `test_syntax_check_is_skipped_for_non_python_files` and
`test_new_non_python_file_with_python_like_garbage_is_still_allowed` lock in that a `.md`/`.txt`
file containing Python-shaped garbage is never blocked by this. If a future language gets its own
deterministic syntax check, give it its own scoped helper the same way rather than trying to
generalize `_validate_python_syntax` into something it isn't.

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
(`git status`, the file on disk) rather than trusting either the model's or your own assumption.

A related but distinct variant, seen live via user reports: instead of fabricating a fake
`<tool_response>`, the model sometimes never attempts a tool call at all — it prints the proposed
new file content or a `diff`-fenced code block as plain chat text and asks "Would you like me to
save this?", and keeps doing so turn after turn even when the user replies "yes"/"proceed", because
each reply just becomes more conversation for it to narrate over rather than a trigger to call
edit_file/write_file. `SYSTEM_PROMPT_TEMPLATE` (`agent/cli.py`) now has an explicit instruction
against this — "call edit_file or write_file immediately... your very next output must be the tool
call itself, not another description" — mirrored for `run_command`/Git in the paragraph right after
the Git section. This materially reduces the failure rate but does not eliminate it; the system
prompt already instructs the model not to do this either way, a small model doesn't always comply.

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

**Interception alone doesn't stop a model from just asking again — `MAX_REPETITION_ESCALATIONS`
bounds that too.** Live testing surfaced the actual gap: `qwen2.5-coder:3b` kept issuing an
identical `git_status()` call *past 10 times in a row* after the first `repetition_detected`
correction fired — each interception is real (no re-execution), but nothing previously stopped the
model from simply requesting the same call again next iteration, so the turn silently burned most
of `MAX_TOOL_ITERATIONS` and ended in an unhelpful `max_iterations` with no answer at all. Once
`consecutive_call_count` has triggered interception `MAX_REPETITION_ESCALATIONS` (2) times in a
row for the same signature (i.e. the 5th identical attempt), the turn now gives up outright —
`{"type": "final", "task_incomplete": True}` — instead of yielding a 3rd `repetition_detected` and
continuing. Same bounded-then-stop shape as `MAX_NARRATION_RETRIES` below, applied to this
different signal; `cli.py`'s `task_incomplete` rendering was generalized to cover both causes
rather than assuming narration. `tests/test_loop.py::TestRepetitionDetection::
test_repetition_gives_up_after_escalation_limit_instead_of_looping_forever` and
`tests/test_stress.py::TestRepeatedToolCallsAtScale::test_twenty_identical_calls_only_execute_twice`
(updated — it previously asserted the old unbounded-looping behavior as if it were correct) cover
this.

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

### Phase 9: context, performance & Qwen 3B optimization

**Before touching this area, know what Phases 1–8 already built**, verified by direct code reading
rather than assumed: `context_manager.py`'s `compact_messages()` already enforces a 12,000-char
whole-conversation budget (`MAX_CONTEXT_CHARS`) with a protected floor of the most recent 6 tool
messages, already retroactively placeholders a superseded duplicate `read_file` or one invalidated
by a later `edit_file`/`write_file`, and `TaskState.summarize()` was already bounded per-category
and regenerated (never appended) into the system prompt every iteration. Every tool output already
had a real cap before Phase 9: `run_command` stdout/stderr (12K chars each), `git_diff` (8K),
`git_log` (≤20 entries), `read_file` (200KB/20MB/800-line tiers), `list_files` (≤400 entries),
`search_files` (≤50 results, 5/file, 200-char previews) — and the model never saw a raw diff at all,
only a one-line confirmation after `edit_file`/`write_file`. Phase 9 did not rebuild any of this; it
closed the gaps a careful read of that existing code actually found.

**What Phase 9 explicitly evaluated and did NOT change, and why:**
- Reordering `messages` by priority category (a literal reading of "consistent context order"):
  `context_manager._iter_tool_messages_with_args` depends on strict chronological
  assistant-tool_calls ↔ tool-result pairing. Reordering the real conversation list would break that
  pairing — a correctness regression far worse than any latency gain. The system prompt's own
  internal section order is where "consistent order" safely applies.
- Rewriting all 16 tool descriptions to be shorter: they're already 200–470 chars and were shaped by
  live-testing Qwen's tool-calling quirks (see the sections above). Shrinking them risks the exact
  reliability regression this phase's own "revert if it hurts reliability" rule exists to prevent.
  Only a short additive tool-selection paragraph was added to the system prompt instead.
- Partial (head+tail) truncation in `compact_messages` pass 2, instead of the existing all-or-nothing
  placeholder: unverified upside against a well-tested, working path — left as-is.

**The unchanged-file read cache (the main new mechanism) reuses two already-correct primitives
instead of inventing new state.** `tools/filesystem.py`'s `_read_file` short-circuits a **full**
(non-ranged) `read_file` call to a short "unchanged since you last read it" notice — instead of
resending the whole file — when: `task_state is not None`, the path is still in
`task_state.files_inspected` (already removed by `note_file_modified()` the moment the agent edits
that file — the existing cache-invalidation contract, not new), and `tracker.is_fresh(path,
current_disk_content)` is true (already-existing SHA-256 comparison against the hash **recorded
before** this read, so it also catches a file changed outside the agent's own edits — e.g. the user
editing it directly — not just edits the agent itself made). A new `force: bool` arg on
`ReadFileArgs` bypasses it. This makes `read_file` the **second** documented, narrow exception to
"tools don't know about `TaskState`" (the first being `update_plan`/`get_plan`) — `task_state` is
threaded into `build_read_file_tool()` in `agent/tools/__init__.py`'s `build_default_registry()`
for exactly this, read-only.

**Why `is_fresh()` must be checked before `tracker.record()`, not after.** `_read_file` computes
`is_cache_hit` from the *previously* recorded hash before calling `tracker.record()` with the
current content. Reversing that order would make `is_fresh()` trivially true right after recording
(you'd be comparing content against its own just-recorded hash), silently breaking external-drift
detection — a file changed outside the agent's own tools would then wrongly short-circuit as
"unchanged." `tests/test_tools_filesystem.py::TestReadFileCache` locks this ordering in directly
(`test_cache_miss_after_external_disk_modification`).

**A cache-hit result never gets confused with a real one downstream.** `ToolResult` gained a new
optional `cache_hit: Optional[bool] = None` field (same additive pattern as Phase 8's
`error_type`/`recoverable` — default `None`, every pre-Phase-9 `ToolResult` call site unaffected).
`loop.py` copies it onto the `messages` dict it appends (`msg["cache_hit"] = True`, a non-standard
extra key on the tool-role dict, same precedent as the existing `tool_name` key Ollama's own schema
doesn't require). **This is what fixes a real bug that would otherwise exist**: `compact_messages`'s
pass 1 supersedes an *older* `read_file` result once a *newer* one for the same path appears — but
if that "newer" one is only a cache-hit stub with no real content, superseding the older real read
would leave the model with **zero copies** of the file anywhere in context. Pass 1 now checks
`msg.get("cache_hit")` and skips superseding (and skips advancing its own bookkeeping index) when
the newer occurrence is a stub — the *real* older read stays live until an actual subsequent real
read supersedes it. `tests/test_context_manager.py::TestCompactMessagesCacheHitAware` and
`tests/test_loop.py::TestReadFileCacheIntegration` cover this end to end, not just at the unit level.

**`compact_messages()` now returns a `CompactionStats` (superseded/stale/trimmed counts) instead of
`None`** — purely additive; every caller that ignored the return value before still works. Used only
for the debug-mode summary below and for tests asserting on what actually got compacted, rather than
just that the resulting message list looks right.

**Debug-mode performance surface, not a new event type.** `run_agent_turn()`'s entire iteration loop
is now wrapped in `try/.../finally`, guaranteeing exactly one `logger.debug(...)` summary line per
turn (context chars/estimated tokens, LLM-call count, tool-call count, cache hits/misses,
superseded/stale/trimmed counts) regardless of which of the many exit paths (final, error,
cancelled, max_iterations) actually fired — this reuses Phase 8's existing `get_logger`/
`configure_logging(debug=...)` plumbing (already redaction-filtered, already gated behind
`--debug`/`CODE_AGENT_DEBUG=1`) rather than inventing a new structured-debug subsystem or a new
event type the VS Code extension/CLI would need to learn about. If you add a fourth "apply"-style
early-return path to `run_agent_turn` in the future, it's automatically covered by the same
`finally` — no separate logging call needed at the new return site.

**Token estimation (`agent/context_budget.py`) is a reporting layer only — it does not replace the
proven char-based enforcement.** `estimate_tokens`/`estimate_tokens_from_chars` are a bare `len(text)
/ 4` heuristic (`CHARS_PER_TOKEN_ESTIMATE = 4`), used only for the debug log line and the
benchmark's `estimated_peak_context_tokens` metric. `compact_messages`'s actual budget enforcement is
still pure character counting, unchanged — don't wire the token estimate into any enforcement path;
it was deliberately kept decorative because the char-based budget is already tuned and tested, and
"approximately right token count for display" and "hard enforcement threshold" have different
accuracy requirements.

**`CODE_AGENT_MAX_CONTEXT_CHARS` follows the exact `OLLAMA_TIMEOUT`/`timeout_from_env()` pattern**
(`context_budget.max_context_chars_from_env()`: tolerant parsing, falls back to
`context_manager.MAX_CONTEXT_CHARS` on a missing/non-numeric/non-positive value). `run_agent_turn()`
gained a `max_context_chars` parameter (default unchanged) threaded straight into its
`compact_messages()` call; `cli.py`/`server.py` each read the env var once at startup (mirroring
exactly how they already read `OLLAMA_TIMEOUT`) and pass it down through `render_turn()`/the
per-request `run_agent_turn()` call respectively.

**`TaskState`'s underlying list storage is now capped at the source (`_MAX_STORED = 50`), not just
`summarize()`'s display slice.** Before Phase 9, `commands_executed`/`errors_encountered`/
`git_operations`/`files_inspected` grew without bound in memory for the life of a long-running task
even though only the last few of each were ever shown — `summarize()`'s existing per-category
display caps (`MAX_FILES_IN_SUMMARY` etc.) are untouched; this only bounds what's held in the Python
list itself.

**`git_status` gained `MAX_STATUS_ENTRIES_PER_CATEGORY` (mirroring `list_files`'s
`MAX_LIST_ENTRIES` pattern exactly)** — it was the one tool output with no cap at all before Phase
9; a huge uncommitted change (thousands of untracked files) would otherwise list every single path.

**The benchmark harness (`agent/benchmark.py`) was extended in place, not replaced.** `TaskResult`
gained `llm_calls`, `peak_context_chars`, `estimated_peak_context_tokens`,
`time_to_first_token_seconds`, `peak_rss_bytes` — all with safe defaults so a pre-Phase-9 saved
report still loads. `_RecordingClient.chat()` already saw `messages` on every call, so per-call
context size and the first-streamed-update timing are computed there with no changes needed to
`loop.py`/`context_manager.py` for those two metrics specifically. The JSON report shape changed
from a bare array to `{"model", "host", "timestamp", "tasks": [...]}` (a real gap: nothing
previously identified which run/model produced a saved report) — `_load_report()` tolerates both
shapes so a genuine pre-Phase-9 baseline file (captured deliberately, per this phase's own
baseline-first requirement, **before** any of these changes were made) still works with the new
`--compare BEFORE.json AFTER.json` mode. Two new tasks were added following the exact same
`_setup_fixture_project`-based pattern as the original six: **Relevant file selection** (a fixture
with deliberately irrelevant noise files — `frontend/`, `package-lock.json`, `vendor/` — checked via
`_NOISE_PATH_MARKERS` against every `tool_call` event's `path` argument, not by inspecting model
prose) and **Stuck/repetition recovery** (a request whose literal `old_text` doesn't exist in the
target file, designed to provoke Phase 8's `MAX_CONSECUTIVE_IDENTICAL_CALLS` repetition-detection
mechanism; the check only requires the turn to recover with a clean final answer rather than hitting
`max_iterations`, regardless of whether `repetition_detected` specifically fired, since a model that
avoids the repeated call in the first place is an equally valid "recovery").

**`peak_rss_bytes` is a process-wide high-water mark (`resource.getrusage(...).ru_maxrss`), not a
clean per-task delta** — reading it after each task still gives a useful monotonically-growing
signal across one benchmark run, but callers displaying it should say so. Units differ by platform:
bytes on macOS (the target hardware), KB on Linux.

**A real, honest baseline exists from before any Phase 9 code changed**, captured per this phase's
own explicit process requirement (baseline first, never claim an improvement the numbers don't
demonstrate): only 1/6 original tasks passed against the live `qwen2.5-coder:3b` setup at the time,
dominated by the model narrating/failing to call tools correctly rather than by anything Phase 9
touches — see the actual before/after numbers reported to the user rather than duplicating them
here, since real model behavior varies run to run and this file should not go stale with one
snapshot's numbers.

### Phase 10: autonomous filesystem operations and tool-calling reliability

Phase 9 closed context/performance gaps; Phase 10 targeted a different, live-observed problem: the
model narrating an intended change ("I'll create sorting.py...", "What would you like me to do
next?") instead of calling a tool, and the agent having no capability at all — not even a
reliability gap — to fulfill a plain "rename this file" or "delete that file" request, since
`command_policy.py`'s `run_command` allowlist deliberately excludes `rm`/`mv` and there was no
structured alternative. Both problems compound on requests like "rename bubble_sort.py to
sorting.py and add the other sorting algorithms," which is why that's the scenario
`tests/test_loop.py::TestEndToEndSortingScenario` exercises end-to-end.

**`delete_file`/`rename_file` are new tools, not a `run_command` allowlist change, by explicit
requirement.** `agent/file_ops.py` (`ProposedFileOp`, `validate_delete`/`validate_rename`) mirrors
`git_policy.py`'s split from `agent/tools/git.py` exactly: pure validation, no filesystem access,
raising `FileOpError` with a model-facing explanation. `agent/tools/file_ops.py` builds the two
tools and `apply_file_op()` — called directly from `loop.py` after approval, with no
`Tool.execute()` safety net, so (like `apply_git_operation`) every failure mode is caught and turned
into a `ToolResult` rather than left to raise. Deliberately file-only, not directory-capable — a
recursive directory delete/move is a meaningfully larger blast radius than anything else this agent
can do, and every real request this was built for (the sorting.py scenario, "rename app_old.py to
app.py") only ever needs a single file. `rename_file` refuses to overwrite an existing destination
(mirrors `write_file`'s existing "already exists → use edit_file instead" precedent) rather than
supporting an implicit overwrite; `apply_file_op` also re-checks the destination immediately before
renaming (mirrors `apply_git_operation`'s re-checked staged-files/branch) so a same-turn race can't
silently clobber a file that appeared after the proposal was built. Both tools reject the project
root itself and any sensitive-pattern path, same as `edit_file`/`write_file`. `FileStateTracker`
already had a `forget(path)` method (added for a since-superseded purpose) that `apply_file_op` now
uses for real — deleting forgets the source; renaming forgets both source and (harmlessly, since
nothing was ever recorded there) destination, so a later stale-edit check against either path has
nothing incorrect to compare against. `TaskState.note_file_removed()` / `context_manager.
record_file_removed()` extend the existing `note_file_modified`-style invalidation pattern for a
path that no longer exists rather than one that just changed. `compact_messages`'s pass 1
(`agent/context_manager.py`) also now marks an earlier `read_file` result stale when the same path
is later `delete_file`d or is the *source* of a `rename_file` (destination is never marked stale —
it was never read under its new name) — a real gap Phase 9 didn't have a reason to close, since
neither tool existed yet.

**Narration-vs-tool-call detection is a heuristic safety net, not a replacement for prompt
engineering.** `_looks_like_unactioned_narration()` (`agent/loop.py`) matches two narrow signal
classes in a model response that produced zero tool calls: a deferring question ("what would you
like me to do next?", "should I proceed?") and a stated-but-unactioned mutating intent ("I'll
create...", "I will delete..."). Deliberately narrow phrase/verb matching, not a broad "contains
'I will'" check — a genuine informational answer can legitimately contain similar phrasing, and
this is safe to be imprecise about specifically because it's bounded (`MAX_NARRATION_RETRIES = 2`)
and gated on `mutating_tool_called_this_turn` being `False` (tracked via `MUTATING_TOOL_NAMES`,
which deliberately excludes read-only inspection tools — the actual live-observed bug was
`read_file` → `read_file[cached]` → narration, i.e. tool calls happened but none of them were a
mutation). A false positive costs at most a couple of extra local-model turns, never corrupts state,
since narration redirection only ever adds a corrective `user`-role message and loops again — it
never fabricates a tool call itself. On the 3rd unactioned response the turn *does* finalize (so the
agent never hangs), but the `"final"` event carries `"task_incomplete": True`, which `cli.py` renders
as a distinct yellow warning rather than a normal answer — the model's own prose is never treated as
proof a mutation happened; only an actual `change_applied`/`file_op_applied`/`command_result`/
`git_operation_applied` event is. This is additive to, not a replacement for, the pre-existing
system-prompt instructions against narration (see "The model can narrate a fake tool result instead
of calling the tool" above) — the prompt reduces how often it happens, this mechanism catches it
when the prompt alone doesn't.

**The cache-hit response was redesigned from a conversational sentence into structured status,
after live evidence it was actively causing the exact failure it was trying to prevent.** The
original Phase 9 message ("`'{path}' is unchanged since you last read it... reuse it`") reads to a
3B model mid-task as "nothing left to do here" rather than "here's the file's status, continue" —
observed live producing exactly the reported bug (agent stops and asks "what would you like me to
do next?" right after a cache hit). `tools/filesystem.py`'s `_read_file` now returns a `path=...
status=unchanged lines=N hash=...` line (structured, not prose) followed by an explicit "the task is
NOT done yet, proceed" instruction. The full file content is deliberately still NOT resent on a
cache hit — that would defeat the whole point of the cache (this is the same real content already
sitting earlier in the conversation, which `compact_messages`'s existing cache-hit-aware pass-1 logic
already keeps alive rather than superseding away, see Phase 9 above, unchanged by this phase) — the
narration-detection mechanism above is the actual backstop if a small model still stalls after a
cache hit despite the clearer wording, not a second copy of the file.

**`MAX_TOOL_ITERATIONS` raised from 10 to 25.** A multi-file request (read the old file, create its
replacement, update a reference, delete the original, run tests) routinely needs 8-10 real tool
calls even when the model behaves perfectly on the first try; add narration-correction retries or
one re-read after an unexpected tool error and 10 was routinely too tight for exactly the kind of
task this agent is meant to complete end-to-end without a new user message (see "PRIMARY OBJECTIVE"
in the spec this phase implements — multi-tool-call sequences must not be artificially cut short).
`tests/test_stress.py`'s `MAX_TOOL_ITERATIONS`-based assertions import the real constant rather than
a hardcoded number, so they scale automatically.

**Ollama generation parameters are now explicitly set, not left at the model's own Modelfile
default.** `OllamaClient` payloads now include `"options": {"temperature": ..., "top_p": ...}`
(`DEFAULT_TEMPERATURE = 0.2`, `DEFAULT_TOP_P = 0.9`, both overridable via `OLLAMA_TEMPERATURE`/
`OLLAMA_TOP_P` following the exact `OLLAMA_TIMEOUT`/`timeout_from_env()` tolerant-fallback pattern).
Previously unset, so Ollama fell back to qwen2.5-coder's own Modelfile default (0.7-0.8 range) —
tuned for open-ended chat, not for reliably emitting well-formed tool-call JSON turn after turn. Not
0.0: a small amount of variance still helps the model recover after a rejected/failed call instead
of deterministically repeating the exact same (wrong) attempt (this is a *different* mechanism from
the repetition-detection guard in "Reliability and failure recovery (Phase 8)" above — that one
forcibly stops an identical repeated call; this is about not making that outcome the deterministic
default in the first place). Deliberately did NOT add a `num_predict` cap despite the spec's general
"avoid unnecessarily high max tokens" guidance — a low cap risks truncating a legitimate
`write_file` call's `content` argument mid-generation (e.g. a file with several functions in it),
which would corrupt the tool-call JSON entirely and produce a strictly worse failure than the
verbosity this would have saved.

**`run_command`'s output compression is scoped to large, fully-passing test runs only.**
`tools/terminal.py`'s `_compress_if_clean_test_run()` collapses stdout to just its final summary
line ("124 passed in 3.20s") when: the command actually succeeded (`exit_code == 0` — a failure is
never compressed, the model needs the real detail to fix it), stdout is over
`TEST_OUTPUT_COMPRESSION_THRESHOLD_CHARS` (2000 chars), and a recognizable pytest/unittest-style
summary line is found near the end with no "failed"/"error" in it (defense in depth on top of the
exit-code check, mirroring the same regex-based summary extraction `context_manager.py`'s
`_summarize_command_outcome` already used for `TaskState`, but applied here to the actual
model-facing tool output rather than only the compact task-memory record). Runs before, not instead
of, the pre-existing `MAX_STDOUT_CHARS`/`MAX_STDERR_CHARS` truncation in `execute_command` — a
genuinely huge stdout still gets truncated first, so the summary-line regex has real content to find
rather than being cut off mid-match (a real bug caught during this phase's own testing: constructing
test fixtures large enough to exceed `MAX_STDOUT_CHARS` before the summary line was reached made the
line itself get truncated away, taken as a lesson to size test fixtures under that limit instead of
raising it).
