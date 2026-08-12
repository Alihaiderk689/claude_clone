# code-agent (Phase 5)

A local, free, Claude Code–style coding assistant that runs entirely on your own
machine using [Ollama](https://ollama.com) and `qwen2.5-coder:3b`. No cloud API,
no API key, no subscription.

This is **Phase 5**: the agent now has Git awareness. It can inspect status,
diffs, and history freely (read-only, no approval needed), and can propose
creating a branch, staging specific files, or committing — but exactly like
file edits and commands, it can never touch Git state on its own. Every
branch/stage/commit is shown to you first and only happens after you
explicitly approve it. There's no generic "run a Git command" tool and never
will be for destructive operations — see "Git integration" below for exactly
what is and isn't possible. Pushing to a remote still doesn't exist. See
"What is NOT implemented yet" below.

## Prerequisites

- macOS (tested on Apple Silicon, e.g. MacBook Air M3, 8 GB RAM)
- Python 3.9+
- [Ollama](https://ollama.com) installed and running locally
- Optional: [ripgrep](https://github.com/BurntSushi/ripgrep) (`brew install ripgrep`)
  for faster code search — a pure-Python fallback is used automatically if it's
  not installed

## 1. Install Ollama

```bash
brew install ollama
```

Or download the installer from [ollama.com/download](https://ollama.com/download).

Start the Ollama server (if it isn't already running as a background service):

```bash
ollama serve
```

## 2. Pull the coding model

```bash
ollama pull qwen2.5-coder:3b
```

This downloads a ~1.9 GB model — chosen over the 7B variant for lower memory
use and faster inference on an 8 GB machine. `qwen2.5-coder` supports Ollama's
native tool calling, which the agent relies on for inspecting, editing, and
running commands in the project. A larger model such as `qwen2.5-coder:7b`
also works (just set `OLLAMA_MODEL` accordingly) and tends to use tools more
reliably, at the cost of more memory pressure on 8 GB and slower generation —
see the live-testing notes further down for the trade-off actually observed.

## 3. Create a virtual environment

From the `code-agent/` directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 4. Install dependencies

```bash
pip install -r requirements.txt
```

Or, to install `code-agent` as a command:

```bash
pip install -e .
```

## 5. Start the application

Run it from inside the project you want the agent to inspect:

```bash
cd my-project
code-agent
```

The current working directory becomes the agent's **project root** — see
"Project-root awareness" below. If installed without `pip install -e .`, run
it as a module instead: `python -m agent.cli`.

## Configuration

Both settings are optional; sensible defaults are used if unset.

| Variable        | Default                 | Purpose                          |
|-----------------|--------------------------|-----------------------------------|
| `OLLAMA_MODEL`  | `qwen2.5-coder:3b`       | Which Ollama model to chat with  |
| `OLLAMA_HOST`   | `http://localhost:11434` | Base URL of the Ollama server    |

Example:

```bash
OLLAMA_MODEL=qwen2.5-coder:3b OLLAMA_HOST=http://localhost:11434 code-agent
```

## Example usage

```
$ cd my-project
$ code-agent

╭──────────────────────────────────────────╮
│ Local Coding Agent                        │
│ Model: qwen2.5-coder:3b                   │
│ Project: /Users/me/projects/my-project    │
╰──────────────────────────────────────────╯

You > Where is JWT authentication implemented?

Agent is inspecting the project...

  → list_files('.')
  → search_files('JWT')
  → read_file('backend/accounts/views.py')

Agent > JWT authentication is implemented in backend/accounts/views.py, inside
GoogleLoginView.post(), which issues a JWT after a Google OAuth login.

You > exit

Goodbye!
```

Plain conversational questions (no project inspection needed) skip the tool
step entirely and go straight to an answer — a "thinking..." spinner runs
while waiting; see "How tool calling works" below for why responses are
shown all at once rather than streamed token-by-token. You can also exit
with `quit`, `Ctrl+C`, or `Ctrl+D`.

### Editing example

```
You > Fix the authentication error so it returns a message instead of None

Agent is inspecting the project...

  → search_files('authentication')
  → read_file('backend/auth.py')

Proposed change:

Modified file: backend/auth.py
────────────────────────────────────────────────────────────
--- a/backend/auth.py
+++ b/backend/auth.py
@@ -1,7 +1,7 @@
 def login(username, password):
     if username == "admin" and password == "1234":
         return {"token": "abc123"}
-    return None
+    return {"error": "Invalid credentials"}
────────────────────────────────────────────────────────────
Apply this change? [y/N/a]: y
✓ Change applied successfully. (backend/auth.py)

Agent > The authentication failure handling has been updated.
```

Answering `n` (or just pressing Enter) leaves the file untouched:

```
Apply this change? [y/N/a]: n
✗ Change rejected. No files were modified.
```

`a` approves this change and every further proposed change for the rest of
the current turn, without asking again.

### Command example

```
You > Run the tests for this project

Agent is inspecting the project...

  → list_files('.')
  → read_file('pyproject.toml')

Agent wants to run:

  pytest

Working directory:
  /Users/me/projects/my-project

Allow this command? [y/N]: y

Running pytest...

Command finished (exit code 1).
--- stdout ---
tests/test_auth.py .F                                                    [100%]
FAILED tests/test_auth.py::test_failed_login_returns_error_dict - AssertionError

Agent > One test is failing: test_failed_login_returns_error_dict expects a
dict with an "error" key, but login() currently returns None on failure.
```

Unlike file edits, there's no `a` shortcut for commands — every proposed
command is confirmed individually, and a bare Enter rejects it just like `n`.

## Project-root awareness

The directory `code-agent` is started in becomes its project root. It's
captured once at startup, shown in the banner, and every tool call is
resolved relative to it — the agent cannot read or list anything outside that
directory, no matter what path a tool call asks for.

## Tools

Read-only inspection:

- **`list_files(path=".", max_entries=200)`** — lists files/directories under
  `path`, skipping irrelevant directories (`.git`, `node_modules`,
  `__pycache__`, `.venv`, `venv`, `env`, `dist`, `build`, `coverage`,
  `.pytest_cache`, etc.) and capping output so a huge repo never floods the
  model's context.
- **`read_file(path, start_line=None, end_line=None)`** — returns
  line-numbered file contents. Full reads are capped at 200 KB; larger files
  must be read in ranges. Directories, binary files, nonexistent files, and
  sensitive files (`.env`, `.env.*`, `*.pem`, `*.key`, `id_rsa*`, credentials
  files, etc.) are all rejected with a clear message instead of a traceback.
- **`search_files(query, path=".", max_results=30)`** — literal, case-sensitive
  text search across the project, returning `path:line: content`. Uses `rg`
  (ripgrep) when it's on `PATH` for speed; otherwise falls back to a small
  pure-Python scanner. Both paths skip ignored directories and sensitive
  files and cap total results.
- **`git_status()`** — current branch (or detached-HEAD state), and a
  structured breakdown of staged/modified/deleted/renamed/untracked files.
  Reports "This project is not a Git repository" gracefully if it isn't one.
- **`git_diff(staged=False, path=None)`** — unified diff of unstaged or
  staged changes, optionally limited to one file/directory. Truncated with a
  clear marker past 8,000 characters.
- **`git_log(limit=10)`** — up to 20 recent commits as `<short-hash> <subject>`
  lines.

Approval-gated editing (see "Safe file editing" below):

- **`edit_file(path, old_text, new_text)`** — proposes replacing an exact,
  unambiguous block of text in an existing file.
- **`write_file(path, content)`** — proposes creating a brand-new file; fails
  if the file already exists.

Approval-gated command execution (see "Safe command execution" below):

- **`run_command(program, args=[], timeout=120)`** — proposes running one
  local development/test command from a small allowlist: `pytest`, `mypy`,
  `ruff check`/`ruff format`, `npm test`/`npm run <script>`, `python -m
  pytest`/`python -m unittest`, `python <project script>.py`, or `node
  <project script>.js`. Anything else — a different program, shell syntax,
  installers, network tools — is rejected before the user is ever asked.
  (`git` itself is explicitly on this tool's denylist too, so there's no way
  to reach Git through it — see "Git integration.")

Approval-gated Git operations (see "Git integration" below):

- **`git_create_branch(name)`** — proposes creating a new branch from HEAD
  (not checked out). Fails if the name is invalid or already exists.
- **`git_stage(paths)`** — proposes staging specific files. Files matching a
  sensitive pattern trigger an explicit warning the user must acknowledge.
- **`git_commit(message)`** — proposes committing the currently staged files.
  Fails if nothing is staged.

New tools plug into this system with the same shape: a Pydantic argument
model, a description, and a run function (see `agent/tools/base.py`).

## Safe file editing

The model never writes to disk. `edit_file` and `write_file` only *propose* a
change — they validate the request and build the exact new file content in
memory, entirely in Python, and hand it back as a `ProposedChange`
(`agent/diff.py`) with no side effects. The agent loop (`agent/loop.py`) then
pauses and shows the user a diff; the file is only written if the user
explicitly approves, via a dedicated `apply_change()` step the model itself
can never trigger. This is the whole point of Phase 3:

```
LLM proposes → Python validates → diff shown → USER APPROVES → Python writes
```

**edit_file** replaces one exact block of text:

- `old_text` must appear in the file **exactly once** — if it's missing or
  ambiguous (appears more than once), the tool refuses and asks the model to
  provide more surrounding context, rather than guessing which occurrence
  was meant.
- **Stale-edit protection**: `read_file` records a hash of whatever it reads;
  `edit_file` checks that hash before proposing a change. If the file
  changed on disk since it was last read (by you, another process, or a
  previous turn), the edit is refused with "changed on disk since it was
  last read" instead of silently clobbering those changes. A path that was
  never read has nothing to check against, so this only ever blocks a
  genuinely stale edit, never a fresh one.

**write_file** creates a new file and refuses if one already exists at that
path — `edit_file` is required for existing files, so the model can't
accidentally overwrite something by picking the wrong tool.

**Applying an approved change** (`apply_change()` in `agent/tools/editing.py`):

- Writes atomically — the full new content goes to a temp file in the same
  directory, `fsync`'d, then swapped into place with `os.replace()`. If
  anything fails before that final swap, the original file is untouched and
  the temp file is cleaned up.
- Preserves the original file's newline style (`\n` vs `\r\n`) and POSIX
  permissions; new files get `0o644`.
- Reads the result back and verifies it matches what was approved before
  reporting success — a silent mismatch is reported as a failure instead.
- Updates the stale-edit tracker with the new content, so a second edit to
  the same file later in the same session isn't wrongly flagged as stale.

**Approval prompt**: `y`/`yes` applies, `n`/`no`/a bare Enter rejects (reject
is always the default — an unanswered prompt never applies anything), and
`a` approves this change plus every other proposed change for the rest of
the current turn. Multiple proposed changes in one turn are always reviewed
one at a time, never silently batch-applied.

**Rejected and failed edits**: a rejection is reported back to the model as
"User rejected this change. The file was not modified." — the model is
instructed not to immediately re-propose the same edit. A failed edit
(target text not found, ambiguous match, stale file) is reported with the
specific reason so the model can `read_file` again and retry with a more
precise edit, rather than the app guessing what was meant.

**Security**: `edit_file`/`write_file` go through the exact same
`ProjectRoot.resolve()` and sensitive-file checks as the read-only tools
(see below) — path traversal, absolute paths outside the root, and
`.env`/key/credential-shaped files are all rejected before a diff is ever
generated, regardless of what the model requests.

## Safe command execution

Same principle as editing, applied to running commands: the model never
executes anything. `run_command`'s `run()` only validates the request
(`agent/command_policy.py`) and builds an `ApprovedCommand` describing
exactly what would run; the agent loop pauses with a `confirm_command` event,
and only an explicit approval leads to `execute_command()`
(`agent/tools/terminal.py`) actually spawning a subprocess.

```
LLM proposes → Python validates → shown to user → USER APPROVES → Python runs it
```

**The executable and every argument are checked structurally** — never just
"does this program exist on `PATH`":

- **Allowlist, not denylist**: only `pytest`, `mypy`, `ruff`, `npm`, `python`,
  `python3`, and `node` are recognized at all; a long list of specific
  dangerous programs (`rm`, `sudo`, `curl`, `wget`, `ssh`, `chmod`, `git`,
  `bash`, ...) is named explicitly for clear rejections, but the real gate is
  that anything not in the allowlist is rejected by default.
- **Per-program argument shape validation**: `ruff`'s first argument must be
  `check` or `format`; `npm`'s subcommand must be `test` or `run <script>`
  (and the script name can't be `install`, `publish`, or similar); `python`/
  `python3` must be `-m pytest`/`-m unittest` or an existing project `.py`
  script (`-c` is always rejected); `node` must be an existing project
  `.js`/`.mjs`/`.cjs` file (`-e`/`-p`/`-r` are always rejected). `pytest`/
  `mypy` accept flags and paths, but every non-flag argument is resolved
  through the same `ProjectRoot.resolve()` the file tools use, so a path
  argument can't point outside the project.
- **Never `shell=True`, ever**: the subprocess is always invoked as
  `[program, *args]`. Because of this, shell metacharacters in an argument
  (`;`, `&&`, `||`, `|`, `>`, `<`, `` $( `` , `` ` ``, `&`) are structurally
  inert — there's no shell to interpret them — but they're still explicitly
  rejected up front as defense in depth (and to give a clear error instead of
  a confusing one from whatever program received them literally). Verified
  live: even when the model split an injection attempt across separate
  argument tokens instead of one string, the result was just `pytest`
  receiving `rm`, `-rf`, `.` as its own inert arguments — `rm` itself was
  never invoked, because `subprocess` only ever executes the first argv
  element as the program.
- **No dependency installation**: `npm install`/`i`/`ci`, `pip`, `brew`, etc.
  are all rejected. If something's missing, the agent is expected to tell you
  and let you install it yourself.
- **No network tools, no background processes**: `curl`/`wget`/`ssh`/`scp`/`nc`
  are rejected outright, and `nohup`/`&`/detachment are rejected as arguments
  too — nothing is allowed to keep running after the command "finishes."

**Timeouts**: every command has one — 120s by default, the model may request
up to 300s, and anything outside 1–300s is rejected. `subprocess.run`'s own
timeout handling kills the process on expiry, so nothing is left running.

**Output limits**: stdout and stderr are each capped at 12,000 characters
before being sent to the model, with an explicit `[stdout truncated because
it exceeded the ... limit]` marker when that happens — the terminal display
is capped more tightly still, to 60 lines per stream, purely for readability
(the model still gets the fuller, still-capped version).

**Environment**: the subprocess receives a stripped-down environment
(`PATH`, `HOME`, `LANG`, `VIRTUAL_ENV`, and a few similar non-secret
variables) rather than a full copy of the agent's own environment — this is
belt-and-suspenders beyond just "don't relay env vars in the result": even a
test that logged `os.environ` wouldn't find whatever tokens or credentials
happen to be set in your shell.

**A nonzero exit code is a normal result, not an error** — the model is
meant to see `pytest` fail, read the output, and propose a fix. Only a
genuine failure to run the command at all (timeout, program not installed,
an OS-level error) shows up as a `tool_error`.

## Git integration

Read-only Git inspection (`git_status`, `git_diff`, `git_log`) needs no
approval — nothing about them can change repository state, so they behave
like `list_files`/`read_file`/`search_files`. The three operations that
*do* change something — creating a branch, staging files, committing — each
get their own narrow tool and follow the exact same propose → show user →
approve → act pipeline as file edits and commands:

```
LLM proposes → Python validates → shown to user → USER APPROVES → Python runs it
```

**There is no generic Git command tool, and there never will be one for
destructive operations.** `agent/git_policy.py` validates only branch names,
staging paths, and commit messages — there's no code path that accepts an
arbitrary Git subcommand or flag string from the model. This is what makes
`git reset --hard`, `git clean -fd`, `git checkout -- .`/`git restore .`,
`git branch -D`, `git rebase`, `git merge`, and `git push`/`--force`
*structurally* unreachable, not just discouraged by the system prompt — none
of them have a tool, so the model has no mechanism to request them. (`git`
is also explicitly denylisted in `run_command`'s policy from Phase 4, so
there's no back door through the generic command tool either.) Verified
live: asked directly to run `git reset --hard`, the model had nothing to
call and declined outright; asked to run `git push --force` via
`run_command`, that was rejected by the allowlist before any approval
prompt appeared.

**Branch names** (`git_create_branch`) are checked against a strict pattern
(letters/digits/`.`/`_`/`-`/`/`, no leading `-`, no `..`, no shell
metacharacters) and then against Git's own `git check-ref-format --branch`
as a second, authoritative check. The branch is created but never checked
out, so it can't disrupt whatever's currently checked out.

**Staging paths** (`git_stage`) go through the exact same `ProjectRoot`
path resolution as every other tool — traversal and absolute paths outside
the project are rejected before Git ever sees them. Paths matching a
sensitive pattern (`.env`, `*.pem`, `*.key`, `credentials.*`, `secrets.*`,
...) aren't silently staged *or* silently blocked — they're flagged, and
the approval prompt becomes an explicit warning ("WARNING: `.env` matches a
sensitive-file pattern and may contain secrets. Are you sure you want to
stage it? [y/N]") that still defaults to reject on a bare Enter, same as
every other prompt in this app.

**Commit messages** (`git_commit`) are passed to `git commit -m <message>`
as a single subprocess argument, never built into a shell string — so
characters like `&`, `;`, or quotes inside a message are just normal commit
message text, not something to over-restrict. Only empty messages and
absurdly long ones (>10,000 characters) are rejected. `git_commit` also
refuses to even propose a commit if nothing is staged, and re-checks the
staged file set immediately before actually committing — if it changed
since the commit was proposed (e.g. another `git_stage` call ran in
between), the commit is refused rather than silently committing a different
set of files than what the user approved.

**Nothing is ever swept in.** `git_stage` only ever stages the exact paths
it was given — there's no `git add -A`/`git add .` equivalent anywhere in
this codebase. This was verified both in tests and live: with `README.md`
and `backend/auth.py` both modified, asking the agent to stage and commit
only the `README.md` change left `backend/auth.py` completely untouched and
unstaged.

## How tool calling works

The model isn't asked to emit arbitrary code or invent its own protocol —
`code-agent` uses **Ollama's native `/api/chat` tool-calling support** (the
`tools` field in the request, `tool_calls` in the response), which
`qwen2.5-coder` advertises support for.

Each user message runs through a loop (`agent/loop.py`):

1. Send the conversation plus the available tool schemas to Ollama.
2. If the model's response includes `tool_calls`, validate the arguments
   (Pydantic) and execute the tool — always against the project root, never
   trusting the model's path as-is.
3. Append the tool's result to the conversation as a `role: tool` message and
   go back to step 1.
4. If the model responds with plain content instead of a tool call, show it
   as the final answer and stop.

This repeats until a final answer arrives or `MAX_TOOL_ITERATIONS` (10) is
hit, which exists purely as a safety net against infinite tool-call loops.

### A real Ollama/qwen2.5-coder compatibility gap, and how it's handled

Verified live against a real local Ollama 0.32.8 server: `qwen2.5-coder:7b`
does not reliably populate the API's structured `tool_calls` field. Its own
chat template (`ollama show qwen2.5-coder:7b --template`) documents
`<tool_call>{"name": ..., "arguments": {...}}</tool_call>` as its
function-calling format, but in practice the model's tool-call attempt often
lands in plain assistant `content` instead — as bare JSON, as tagged JSON, or
(observed repeatedly) as free-form reasoning prose with the JSON call
appended at the end, with no tag at all.

Rather than "faking" tool calling or inventing a new protocol, `agent/loop.py`
recognizes that same model-documented shape as a fallback whenever Ollama
doesn't structure it: it scans the full response text for an embedded
`{"name": ..., "arguments": {...}}` object (inside `<tool_call>` tags if
present, anywhere in the text if not) and normalizes it into the same
internal shape a proper `tool_calls` response would produce. The raw JSON (or
the reasoning prose around it) is never shown to the user as if it were the
answer.

One consequence: because a tool call can be buried anywhere in the text with
no reliable early signal, each model response is fully buffered before
display rather than streamed token-by-token — there's no safe way to show
partial text without risking a raw tool call leaking onto the screen. A
"thinking..." spinner covers each wait instead. If a future Ollama/model
combination reports `tool_calls` reliably, this fallback simply never
triggers and this constraint goes away on its own.

## Context management

Only tool *results* are added to the conversation — the whole repository is
never dumped into the model. Directory listings and search results are
capped, large files must be read in line ranges, and command output is
capped too (see "Safe command execution"). This matters on an 8 GB machine
running a local model: keeping tool output small keeps both memory pressure
and generation time manageable.

## Security

The agent still cannot: delete files; read or write files outside the
project root; or read/create/edit `.env`/key/credential-shaped files even
inside the root (including staging them into Git without an explicit
warning-and-override). It cannot write *any* file, run *any* command, or
change *any* Git state, anywhere, without a human explicitly approving it
first — the model only ever proposes. Approved commands are restricted to a
small allowlist with no shell, no network access, no installers, and no
background processes (see "Safe command execution"), and Git is restricted
to three narrow, validated operations with no generic command tool at all
and no way to reach a destructive one (see "Git integration").

None of this depends on the model behaving — it's enforced in Python.
Every tool call's arguments are validated (Pydantic, plus an explicit policy
layer for commands and Git operations), every path is resolved through
`ProjectRoot.resolve()` (`agent/project.py`), which rejects `..` traversal,
absolute paths outside the root, and symlinks that escape it, and every
write, command execution, or Git operation goes through its own propose →
show user → approve → act pipeline with no shortcut back to the model. This
was verified live, not just in tests: a real local `qwen2.5-coder:7b`
genuinely attempted to edit `.env` and to write outside the project root
during Phase 3 manual testing; `qwen2.5-coder:3b` (the default since Phase 4)
genuinely attempted `sudo reboot`, `curl example.com`, and a `run_command`
call with a literal shell-injection string as an argument, and — new in
Phase 5 — genuinely attempted `git push --force` via `run_command` and had
no tool available at all to even attempt `git reset --hard`. Every one of
these was rejected (or simply had no mechanism to run) before any diff or
approval prompt ever appeared.

## What is implemented now (Phase 5)

Everything from Phases 1–4, plus:

- `git_status`, `git_diff`, `git_log` — read-only, no approval needed
- `git_create_branch`, `git_stage`, `git_commit` — approval-gated, see
  "Git integration"
- `agent/git_policy.py`: branch-name validation (regex + `git
  check-ref-format`), staging-path validation (reuses `ProjectRoot.resolve()`),
  commit-message validation — completely separate from `agent/tools/git.py`'s
  execution, mirroring the command_policy.py/terminal.py split from Phase 4
- An explicit warning-and-override prompt when staging a file that matches a
  sensitive pattern, rather than silently staging or silently blocking it
- A stale-commit check: the staged file set is re-verified immediately before
  actually committing, and the commit is refused (not silently redirected) if
  it changed since the commit was proposed
- The agent loop's existing `send()`-based pause extended with a third
  confirmation kind, `confirm_git_operation`, reusing the same generator
  protocol as file edits and commands rather than a new mechanism
- An updated system prompt: inspect the staged diff before proposing a
  commit, never skip straight to committing just because the user said "fix
  and commit it," never claim a branch/stage/commit happened until the tool
  confirms it, and never improvise a reset/clean/checkout-to-discard to
  "resolve" unrelated changes
- Tests for the policy layer (branch-name/path/message validation, including
  shell-metacharacter and traversal rejection), the tool layer (all six Git
  tools against real temporary Git repositories — clean/modified/staged/
  deleted/renamed/detached-HEAD status, diff truncation, log limits, branch/
  stage/commit proposal and execution, the stale-commit check, a structural
  check that no generic Git-command tool exists), and the full propose →
  confirm → execute/reject loop including the mandatory "staging one file
  must never sweep in an unrelated modified file" case
- Live-verified end-to-end against a real local Ollama server and
  `qwen2.5-coder:3b`, in a real Git repository: `git_status` on a clean repo,
  accurate analysis of a real `git_diff`, a genuinely created (and
  not-checked-out) branch, genuine staging and commit with the exact
  target-spec UX, the mandatory unrelated-changes-protected case (verified
  with real `git diff --staged`/`git diff` afterward, not just the model's
  word for it), the sensitive-file staging warning correctly blocking `.env`
  on rejection, `git reset --hard` having no tool for the model to even
  attempt, and `git push --force` via `run_command` rejected by the Phase 4
  allowlist before any approval prompt

## What is NOT implemented yet (future phases)

- Deleting files (`delete_file`)
- Pushing to a remote (`git push`) or any other remote/GitHub/GitLab operation
- Multi-step planning beyond the tool-call loop
- Persistent memory across sessions
- VS Code or other editor integration
- Any additional CLI commands beyond `exit` / `quit`
- Batched upfront review of multiple proposed changes/commands (each is
  currently reviewed as it's proposed, not pre-planned as a numbered set)
- Installing dependencies automatically, running arbitrary scripts beyond the
  allowlisted shapes, background/long-running processes (e.g. dev servers)

## Running tests

```bash
pip install -r requirements.txt
pytest
```

All Ollama calls and tool execution are mocked/use temp directories, so the
full suite runs without Ollama installed or running.

## Project structure

```
code-agent/
├── agent/
│   ├── __init__.py
│   ├── cli.py               # terminal UI: banner, input loop, event rendering,
│   │                         # diff display, file/command/Git approval prompts
│   ├── loop.py               # the tool-calling agent loop (incl. confirm/apply
│   │                         # for file changes, commands, and Git operations)
│   ├── diff.py                # ProposedChange + unified diff generation
│   ├── command_policy.py      # run_command's allowlist-based validator
│   ├── git_policy.py          # branch/staging-path/commit-message validator
│   │                         # + ProposedGitOperation
│   ├── ollama_client.py      # Ollama /api/chat communication (streaming + tools)
│   ├── project.py            # ProjectRoot: path resolution & traversal protection
│   └── tools/
│       ├── __init__.py       # builds the default tool registry
│       ├── base.py           # Tool / ToolResult / ToolError
│       ├── registry.py       # ToolRegistry: schemas + dispatch
│       ├── filesystem.py     # list_files, read_file
│       ├── search.py         # search_files (ripgrep + fallback)
│       ├── editing.py         # edit_file, write_file, apply_change (atomic write)
│       ├── state.py           # FileStateTracker for stale-edit detection
│       ├── terminal.py        # run_command tool + execute_command (subprocess)
│       └── git.py             # git_status/diff/log + git_create_branch/stage/
│                             # commit + apply_git_operation
├── tests/
│   ├── test_ollama_client.py
│   ├── test_project.py
│   ├── test_tools_filesystem.py
│   ├── test_tools_search.py
│   ├── test_tools_editing.py
│   ├── test_tools_terminal.py
│   ├── test_tools_git.py
│   ├── test_command_policy.py
│   ├── test_git_policy.py
│   ├── test_diff.py
│   ├── test_cli_approval.py
│   └── test_loop.py
├── requirements.txt
├── pyproject.toml
├── .gitignore
└── README.md
```
