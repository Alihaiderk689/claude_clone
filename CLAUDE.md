# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A local, free, Claude Code–style coding assistant CLI (`code-agent`) that runs entirely against a
local [Ollama](https://ollama.com) server and `qwen2.5-coder:3b` (default; `qwen2.5-coder:7b` also
works, see README) — no cloud API, no key. It has read-only project inspection tools (including
read-only Git inspection), approval-gated file-editing tools, one approval-gated command-execution
tool, and three approval-gated Git tools (branch/stage/commit); the model can never write to disk,
run a process, or change Git state without going through an explicit user approval. See README.md
for the full user-facing walkthrough, tool list, and phase-by-phase feature history.

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
```

`OLLAMA_HOST` (default `http://localhost:11434`) and `OLLAMA_MODEL` (default `qwen2.5-coder:3b`)
are the only configuration, set as environment variables.

## Architecture

### Event-driven core, terminal-agnostic

`agent/loop.py`'s `run_agent_turn()` is a generator that drives one user turn (model call → optional
tool execution → repeat until a plain-text answer) and *yields structured event dicts* rather than
printing anything. `agent/cli.py` is the only module that imports `rich` and does terminal I/O; it
just consumes those events. Keep that split — new UIs (or a future non-interactive mode) should be
able to drive `run_agent_turn` without touching `cli.py`.

### The propose → confirm → apply pipeline (the core mechanism, used three times)

`edit_file`/`write_file` (`agent/tools/editing.py`), `run_command` (`agent/tools/terminal.py`), and
`git_create_branch`/`git_stage`/`git_commit` (`agent/tools/git.py`) never touch disk, spawn a
process, or run Git themselves. Each tool's `run()` only validates and builds a description of the
pending action in memory — `ProposedChange` (`agent/diff.py`), `ApprovedCommand`
(`agent/command_policy.py`), or `ProposedGitOperation` (`agent/git_policy.py`) — set on
`ToolResult.pending_change` / `.pending_command` / `.pending_git_operation` respectively. When
`run_agent_turn` sees any of those fields set, it does `approved = yield {"type":
"confirm"/"confirm_command"/"confirm_git_operation", ...}` — a real pause using Python's generator
`send()` protocol. `cli.py` renders the diff/command/Git operation, prompts, and resumes the
generator with `gen.send(approved)`. Only a `True` from that send leads to `apply_change()` (atomic
temp-file write + fsync + verify-by-read-back), `execute_command()` (`subprocess.run([program,
*args], shell=False, ...)`), or `apply_git_operation()` (`git branch`/`git add`/`git commit`, also
never through a shell) actually acting. There is no code path from the model straight to disk, to a
subprocess, or to Git — a new mutating/executing tool must follow this same propose/apply split, not
act directly in its `run()`.

Because of this, `cli.py`'s event loop can't be a plain `for event in run_agent_turn(...)` — it uses
`gen.send(value)` manually so it can answer `confirm`/`confirm_command`/`confirm_git_operation`
events (see `render_turn`). Note the asymmetry: file-edit approval supports `y`/`n`/`a` (`a` =
approve all remaining changes this turn); command and Git approval are strictly `y`/`n` with no
batch-approve, by design — see spec/README. Staging a file matching a sensitive pattern additionally
swaps the normal "Stage these files?" prompt for an explicit "WARNING: ... Are you sure ...?" one
(still defaults to reject) — see `_handle_git_confirm` in `cli.py`.

`command_policy.py`/`git_policy.py` (validation, allowlists, argument-shape rules, injection checks
— never execute anything) and `agent/tools/terminal.py`/`agent/tools/git.py` (execution only, given
an already-validated `ApprovedCommand`/`ProposedGitOperation` — never decide whether something is
allowed) are deliberately separate modules in both cases; don't blur that line when extending either.

**There is deliberately no generic "run a Git command" tool, and there must never be one for
destructive operations** (`reset`, `clean`, `checkout -- .`/`restore .`, `branch -D`, `rebase`,
`merge`, `push`). `git_policy.py` only knows how to validate a branch name, a list of staging paths,
and a commit message — there is no function anywhere that accepts an arbitrary Git subcommand
string. `git` is also explicitly in `command_policy.py`'s `DENYLISTED_PROGRAMS`, so there's no back
door through `run_command` either. If a future phase adds `git push`, it needs its own narrow,
validated tool (remote name/branch only, no arbitrary refspec) — not a generic executor.

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
(`.env`, `*.pem`, `*.key`, `id_rsa*`, ...) that `is_sensitive()` blocks for both reading and writing.
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
