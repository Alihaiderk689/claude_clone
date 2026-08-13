# code-agent (Phase 8)

A local, free, Claude Code–style coding assistant that runs entirely on your own
machine using [Ollama](https://ollama.com) and `qwen2.5-coder:3b`. No cloud API,
no API key, no subscription.

This is **Phase 8**: no new user-facing features. Phases 1–7 built the
agent's capabilities (inspection, editing, terminal, Git, planning, VS
Code); Phase 8 makes all of that **reliable** — a failing tool, a dropped
Ollama connection, a malformed model response, a stuck loop, or a user
interruption no longer risks crashing the agent or corrupting its state.
The guiding shape is:

```
User -> Agent -> Tool -> Tool failure -> Agent receives a structured,
classified failure -> Agent analyzes it -> retries / tries something else /
asks the user -> Continues the task
```

not:

```
Tool failure -> Application crashes
```

See "Reliability and Failure Recovery" below for the full picture, and
"What is NOT implemented yet" for what's still out of scope (unchanged from
Phase 7 — this phase added no new capabilities on purpose).

Phase 7 made the same agent reachable from inside VS Code, through a small
local HTTP server (`code-agent serve`) and a VS Code extension
(`vscode-extension/`). The terminal CLI (`code-agent`, no arguments) is
unchanged and still works exactly as before — VS Code is an additional
interface onto the same Python agent, not a replacement for it. See "VS Code
Integration" below for how it works.

Phase 6 added multi-step task tracking: for a task that genuinely spans
several files or concerns, the agent proposes a numbered plan, tracks each
step's status as it works, and keeps a concise running memory of what it's
inspected, changed, run, and hit errors on — without resending full file
contents and terminal output on every single turn, which matters on 8 GB of
RAM with a 3B model. None of this weakens any existing safety guarantee: a
plan is just a checklist, and every file edit, command, and Git operation it
leads to still needs its own separate approval exactly as before. See
"Planning and task state" below for how it works.

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

To use the agent from VS Code instead of (or alongside) the terminal, start
the local agent server from the same project directory:

```bash
cd my-project
code-agent serve
```

and install the extension in `vscode-extension/` — see "VS Code
Integration" below for the full picture and `vscode-extension/README.md`
for install steps. The `code-agent` (no arguments) command above is
completely unaffected by this and keeps working exactly as it always has.

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
with `quit`, `Ctrl+C`, or `Ctrl+D`, and start a fresh task with `/new` (see
"Planning and task state" below) without restarting the whole app.

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
  sensitive files (`*.pem`, `*.key`, `id_rsa*`, credentials files, etc.) are
  all rejected with a clear message instead of a traceback. `.env`/`.env.*`
  is deliberately *not* in this list — see "Safe file editing" below.
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

Planning (see "Planning and task state" below):

- **`create_plan(goal, steps)`** — proposes a numbered plan (2–12 steps) for
  a task that spans several files/concerns. Only proposes it — the user is
  shown the plan and asked to proceed, though (unlike every other approval
  in this app) a bare Enter here means yes, since a plan has no side effects
  of its own.
- **`update_plan(step_id, status, note=None)`** — updates one step's status
  (`pending`/`in_progress`/`completed`/`blocked`/`failed`) on the already-
  approved plan. No approval needed — it only changes in-memory bookkeeping.
- **`get_plan()`** — shows the current plan and each step's status. No
  approval needed.

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
key/credential-shaped files (`*.pem`, `*.key`, `id_rsa*`, `secrets.*`,
etc.) are all rejected before a diff is ever generated, regardless of what
the model requests. `.env`/`.env.*` is the one deliberate exception: by
request, the agent can create and edit it directly like any other file
(e.g. "make the API and put the secrets in `.env`") — everything else
about the approval flow still applies, so nothing is written until you
approve the diff. Staging a `.env` file into Git still triggers the
explicit sensitive-file warning below, independently of this — see "Git
integration."

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
like `list_files`/`read_file`/`search_files`. The operations that *do*
change something — creating a branch, staging files, committing, pushing —
each get their own narrow tool and follow the exact same propose → show
user → approve → act pipeline as file edits and commands:

```
LLM proposes → Python validates → shown to user → USER APPROVES → Python runs it
```

**There is no generic Git command tool, and there never will be one for
destructive operations.** `agent/git_policy.py` validates only branch names,
staging paths, commit messages, and remote names — there's no code path
that accepts an arbitrary Git subcommand or flag string from the model.
This is what makes `git reset --hard`, `git clean -fd`, `git checkout --
.`/`git restore .`, `git branch -D`, `git rebase`, `git merge`, and
`git push --force` *structurally* unreachable, not just discouraged by the
system prompt — none of them have a tool, so the model has no mechanism to
request them. (`git` is also explicitly denylisted in `run_command`'s
policy from Phase 4, so there's no back door through the generic command
tool either.) Verified live: asked directly to run `git reset --hard`, the
model had nothing to call and declined outright; asked to run
`git push --force` via `run_command`, that was rejected by the allowlist
before any approval prompt appeared.

**`git_push` is the one exception to "no push," added later by explicit
request** — it exists precisely because it's narrow, not despite it: it
only pushes whatever branch is *currently checked out* (there's no
checkout tool, so this is never ambiguous) to an *already-configured*
remote (there's no `git remote add` tool either, so the model can't invent
or redirect one — `validate_remote_name` checks the format, and
`agent/tools/git.py` separately confirms the name is one `git remote`
actually lists before ever proposing anything). The approval prompt shows
the destination (remote name + URL) and a preview of the actual commits
that would be sent, generated from a real `git log <remote>/<branch>..
<branch>` (falling back to the branch's own log if the remote branch
doesn't exist yet). The underlying call is a single, fixed
`git push --set-upstream <remote> <branch>` — there is no argument path,
here or in `git_policy.py`, that can add `--force`/`-f`, and a dedicated
test (`test_push_argv_never_contains_a_force_flag`) spies on the real argv
passed to `git` to confirm it. The branch being pushed is re-checked
immediately before the push actually runs (mirroring `git_commit`'s
staged-file re-check) — if you checked out a different branch between
proposing and approving, the push is refused rather than silently pushing
whatever's now checked out. A rejected push (e.g. the remote has commits
you don't have locally) is reported with Git's own error text, never
retried automatically, and never force-pushed to "fix" it. Verified live
against a real local bare-repo remote and a real `qwen2.5-coder:3b`
response: the model correctly called `git_push` from a plain "push the
current branch to origin" request, the approval prompt showed the real
pending commit, and the commit landed on the remote only after explicit
approval.

**Branch names** (`git_create_branch`) are checked against a strict pattern
(letters/digits/`.`/`_`/`-`/`/`, no leading `-`, no `..`, no shell
metacharacters) and then against Git's own `git check-ref-format --branch`
as a second, authoritative check. The branch is created but never checked
out, so it can't disrupt whatever's currently checked out.

**Staging paths** (`git_stage`) go through the exact same `ProjectRoot`
path resolution as every other tool — traversal and absolute paths outside
the project are rejected before Git ever sees them. Paths matching a
sensitive pattern (`*.pem`, `*.key`, `credentials.*`, `secrets.*`, ...) —
plus `.env`/`.env.*` specifically, checked independently via
`looks_like_env_file()` even though `.env` is no longer in the general
sensitive-file list (see "Safe file editing" above) — aren't silently
staged *or* silently blocked. They're flagged, and the approval prompt
becomes an explicit warning ("WARNING: `.env` matches a sensitive-file
pattern and may contain secrets. Are you sure you want to stage it?
[y/N]") that still defaults to reject on a bare Enter, same as
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

## Planning and task state

For a request like "add JWT authentication" — genuinely spanning several
files and concerns — the agent can propose a structured plan before making
any changes, then track progress against it as it works:

```
You > Add password reset functionality

I'll handle this in the following steps:

  ○ 1. Inspect current authentication architecture
  ○ 2. Inspect the user model
  ○ 3. Add a password reset endpoint
  ○ 4. Add email/token handling
  ○ 5. Add tests
  ○ 6. Run the test suite

Proceed? [Y/n]:
```

A bare Enter here means **yes** — the one deliberate exception to every
other approval prompt in this app, because a plan has no side effects of
its own. Approving it is not permission to skip anything else: every file
edit, command, and Git operation the plan leads to still needs its own
separate, default-*reject* approval exactly as in Phases 3–5.

The plan is real structured data (`agent/planner.py`'s `Plan`/`PlanStep`,
not prose the model has to remember), created by `create_plan` and updated
step-by-step with `update_plan` as the agent marks each one `in_progress`
then `completed` (or `blocked`/`failed` with a short reason if it can't
continue). Progress re-renders after every `update_plan` call:

```
  ✓ 1. Inspect current authentication architecture
  ✓ 2. Inspect the user model
  → 3. Add a password reset endpoint
  ○ 4. Add email/token handling
  ○ 5. Add tests
  ○ 6. Run the test suite
```

**Small requests never get a plan.** The system prompt tells the model to
call `create_plan` only for multi-step work and to just act directly for
something like "rename this variable" — `create_plan` itself also rejects
a proposal with fewer than 2 steps, so a trivial "plan" can't be forced
through even if the model tries.

**`agent/task_state.py`'s `TaskState`** is the agent's short-term memory for
the *current* task only — the goal, the active plan, which files it has
inspected or modified, a short log of commands run (with outcomes, not raw
output) and Git operations taken, and unresolved errors. It lives only as
long as the running process; there's no database, no embeddings, nothing
written to disk. See "Context management" below for how this gets used.

**`/new`** clears the plan, task memory, file-read cache, and conversation
history, starting a clean task — it never touches project files or Git
history:

```
You > /new
Started a new task. Project files and Git history are untouched.
```

**Interrupting a task** (Ctrl+C) preserves whatever task state exists so
far instead of just silently dropping it:

```
(response interrupted)

Task paused.

Plan:

  ✓ 1. Inspect current authentication architecture
  → 2. Inspect the user model
  ○ 3. Add a password reset endpoint
  ...
```

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

Phase 6 adds a second layer on top, in `agent/context_manager.py`, since a
long multi-step task can otherwise accumulate a lot of full tool output
across many turns:

```
Tool result -> extracted into TaskState (compact) -> compacted conversation -> LLM
```

**Recording**: after every tool call, a short structured entry — not the
full output — is added to `TaskState`: a file path for `read_file`, a path
for a successful `edit_file`/`write_file`, a one-line outcome for
`run_command` (parsed from pytest's own summary line where possible, e.g.
"18 passed, 2 failed"; a nonzero exit or timeout also becomes a short
recorded error, and a fully-passing test run clears out older recorded
errors since they're no longer relevant), and a line for a successful Git
operation.

**Injecting memory**: at the top of every model round-trip,
`refresh_system_prompt` *regenerates* (never appends to) a "Current task
memory" section on the system prompt from `TaskState.summarize()` — a
bounded block (recent files, recent commands, recent errors, plan progress)
that stays roughly the same size no matter how long the task runs, since
older entries fall off rather than accumulating.

**Compacting the raw conversation**: also every round-trip,
`compact_messages` mutates the actual message history sent to Ollama:

- An older `read_file` result for a path that gets read again later is
  replaced with a short "superseded" placeholder — no need to keep two full
  copies of the same file in context.
- A `read_file` result for a path that was *edited* afterward is replaced
  with a "stale" placeholder instead, so the model is never left reasoning
  from content that no longer matches disk (see "Cache invalidation" below).
- Only if the conversation is still over a size budget (12,000 characters,
  chosen for a 3B model on 8 GB RAM) after that: the oldest tool output —
  never the most recent 6 tool messages, never the system prompt — is
  trimmed to a short placeholder too, oldest first, until back under budget.

None of this touches user or assistant messages, and none of it discards
information the model still needs for its current step — only content
that's superseded, stale, or old enough that the bounded task-memory
summary already covers what mattered about it.

### How file-context caching works

There's no separate cache to query — `read_file` behaves exactly as it did
in Phase 2 (always returns real, current content; see `agent/tools/state.py`'s
`FileStateTracker`, unchanged, for the actual stale-*edit* safety check). The
"caching" Phase 6 adds is about what stays *visible in conversation history*:
`TaskState.files_inspected` is a simple ordered, deduplicated list — reading
the same unchanged file again doesn't grow it, it just moves that path to
the end — and `compact_messages` uses the conversation itself (matching each
`read_file`/`edit_file`/`write_file` call to its real arguments by walking
the message order, not by parsing any tool's output format) to decide which
*older* copies of a file's content are safe to fold away.

### Cache invalidation

If `edit_file` (or `write_file`) succeeds for a path, two things happen:
`TaskState.note_file_modified` removes that path from `files_inspected` (so
the summary no longer implies it's still "known-current") and
`compact_messages` rewrites any *earlier* `read_file` result for that same
path into `"[stale -- backend/auth.py was modified after this read; ...
read_file it again if you need its current contents]"`. A read that happens
*after* the edit is left completely alone. This is a belt-and-suspenders
measure on top of the real safety mechanism: `edit_file` itself already
refuses to propose a change against a file that drifted since it was last
read (Phase 3's stale-edit protection, via `FileStateTracker`) — this layer
is about keeping the model's own view of the conversation honest, not about
the write-safety guarantee itself.

## VS Code Integration

Everything in this section is additive: it wraps the same `run_agent_turn()`
loop, the same tool registry, and the same approval mechanism the CLI
already uses, behind a local HTTP server. No agent logic was rewritten or
duplicated to build this.

```
VS Code Extension  --(local HTTP, 127.0.0.1 only)-->  code-agent serve
   (Webview UI)                                          |
                                                    Agent Loop / Planner /
                                                    Task State / Context
                                                    Manager / Security /
                                                    Filesystem, Editing,
                                                    Terminal, Git tools
                                                           |
                                                        Ollama (qwen2.5-coder:3b)
```

### Starting the server

From inside the project you want the agent to work on:

```bash
code-agent serve
```

```
Local Coding Agent -- server mode
  Server URL:    http://127.0.0.1:8765
  Model:         qwen2.5-coder:3b
  Ollama status: connected
  Auth token:    /Users/you/.code-agent/server.json (chmod 600)

Waiting for VS Code...
```

- Binds **only** to `127.0.0.1` — this is a hardcoded constant in
  `agent/server.py` (`BIND_HOST`), not something any flag or environment
  variable can change, so it is never reachable from the LAN or internet.
- If port 8765 is already in use, it falls back to an OS-assigned port
  automatically; the actual bound port is always written to the token file
  below, so nothing needs manual reconfiguration.
- On startup it writes `~/.code-agent/server.json` (`chmod 600`) containing
  `{"host", "port", "token"}` — a fresh random token generated with
  `secrets.token_hex` every run, never hardcoded anywhere in this repo or in
  the extension source. Every request other than `GET /health` must present
  it as `Authorization: Bearer <token>`; a missing or wrong token gets `401`.
- One server process can serve multiple VS Code windows/workspaces at once
  (see "Workspace isolation" below) — you don't need to start a new one per
  project unless you want to.
- `Ctrl+C` (or a plain `kill`, handled the same way via a `SIGTERM` handler)
  shuts it down cleanly and removes the token file.

### What the server does not do

It does not reimplement the agent. `agent/server.py` builds each
workspace's session with the exact same `_fresh_session_state()` helper
`cli.py`'s own `/new` command uses, and drives turns with the exact same
`run_agent_turn()` generator `cli.py`'s `render_turn()` drives — the HTTP
layer only adds request/response plumbing, session isolation, and auth
around that existing core. Every security boundary from the sections above
(project-root confinement, sensitive-file protection, the `run_command`
allowlist, the Git tool allowlist, and edit/command/Git approval) applies
exactly as it does in the terminal, because it's the same Python code
enforcing it — the VS Code extension is never trusted as a security
boundary and cannot bypass any of it.

### HTTP API

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | `{status, backend, model, ollama_host, ollama_connected}` — no token required, so the extension can distinguish "agent not running" from "agent running." |
| `/chat` | POST | `{workspace_root, message}` → a chunked NDJSON stream of the same events `render_turn()` renders in the terminal (`tool_call`, `confirm`, `confirm_command`, `confirm_git_operation`, `confirm_plan`, `content`, `final`, `error`, `cancelled`, ...). When a `confirm*` event is emitted, the request stays open, paused inside the generator, until a matching `/chat/confirm` call arrives. |
| `/chat/confirm` | POST | `{workspace_root, approved}` — resumes the paused turn above with the user's decision. |
| `/task/status` | GET | `?workspace_root=...` → the current goal, plan, files inspected/modified, recent commands, errors, and Git actions for that workspace. |
| `/task/stop` | POST | `{workspace_root}` — sets that workspace's cancellation flag; a turn in progress (or paused on an approval) stops at the next safe point and yields `{"type": "cancelled"}`. |
| `/task/new` | POST | `{workspace_root}` — same as the CLI's `/new`: resets the conversation, plan, and task memory for that workspace. Does not touch project files or Git history. |

### Cancellation (Stop Task)

`OllamaClient.chat()`/`chat_stream()` and `run_agent_turn()` both gained an
additive, optional `cancel_event: threading.Event` parameter (default
`None` — zero effect on any existing caller, including the CLI, which never
passes one). When set, streaming from Ollama stops within one line of the
response and raises `OllamaCancelledError`; `run_agent_turn()` also checks
it at the top of every iteration, so a turn paused on an approval prompt
that the user never answers still stops promptly once `/task/stop` pushes a
rejection into the waiting queue. In-flight subprocess commands are not
force-killed by Stop Task specifically — they still rely on Phase 4's
existing hard per-command timeout, a deliberate, documented limitation
rather than an oversight.

### Authentication

Modeled on how Jupyter authenticates local dev servers: a random token,
written to a local file only the current user can read
(`~/.code-agent/server.json`, `chmod 600`), regenerated every time the
server starts. The extension reads this file rather than storing or
hardcoding a secret. If you restart `code-agent serve`, reload the VS Code
window (or just reopen the chat panel) so the extension re-reads the new
token.

### Workspace isolation

Sessions are keyed by the resolved absolute path of `workspace_root`
(`agent/server.py`'s `SessionStore`). Two different projects get two
completely separate `ProjectRoot`, `ToolRegistry`, `FileStateTracker`,
`TaskState`, and conversation (`messages` list) — nothing is shared, so
nothing from one project's files, task state, or conversation can leak into
another's. Each session also has its own lock, so a long-running turn in
one workspace never blocks requests for a different one; a second `/chat`
call *for the same workspace* while one is already in flight gets `409`
rather than being silently queued or mixed in.

### The VS Code extension

Lives in `vscode-extension/` as a separate TypeScript project — it is
**not published to the Marketplace**; install it locally from a `.vsix`
(see `vscode-extension/README.md`). It contains no agent logic: `src/client.ts`
only speaks HTTP/NDJSON to the endpoints above, `src/panel.ts` only renders
what the server sends into a native Webview panel and forwards clicks back,
and `src/diffProvider.ts` only visualizes a proposed change's already-sent
old/new content in VS Code's native diff editor (`vscode.diff`) — it never
writes to disk itself.

Commands (Command Palette and, where noted, editor context menu):

- **Local Coding Assistant: Open Chat** — opens the chat panel for the
  current workspace folder.
- Select code, right-click → **Ask Local Coding Assistant** — opens the
  chat with the selection pre-filled as context for you to finish typing.
- **Local Coding Assistant: Explain Selection** / **Fix Selection** — same
  idea, sent automatically; Fix Selection's prompt explicitly tells the
  agent to inspect the surrounding file rather than blindly rewriting only
  the selected lines, and every resulting edit still goes through the same
  diff/approve/apply pipeline as anything else.
- **Local Coding Assistant: Stop Task** / **New Task** — call `/task/stop`
  and `/task/new` respectively; New Task also clears the panel's
  conversation view.

In the chat panel: a ● Connected / ○ Offline indicator and the configured
model name (both read from `/health`, never hardcoded); sub-action
checkmarks under each agent turn (e.g. "✓ Read auth.py"); proposed edits
show **View Diff** (native diff editor) plus **Approve**/**Reject**;
proposed commands show **Run**/**Reject** with the working directory;
proposed Git operations show **Stage**/**Commit**/**Create Branch** plus
**Cancel**; file references like `backend/auth.py:42` in agent replies are
clickable and open that file at that line; and a task-progress panel
mirrors the current plan and modified-files list, refreshed after each
relevant event via `/task/status`.

## Reliability and Failure Recovery

Everything here is additive on top of Phases 1–7's architecture — no
existing tool, the agent loop's core control flow, or any security
boundary was rewritten to build this; failure handling was added around
them.

### Error classification

Every tool failure now carries a machine-readable `error_type` and
`recoverable` flag alongside its human-readable message (`ToolResult` in
`agent/tools/base.py`). A small taxonomy of `ToolError` subclasses
(`ValidationFailedError`, `NotFoundError`, `PermissionDeniedError`,
`StaleStateError`, `ToolTimeoutError`, `ExternalToolUnavailableError`) is
used at the sites where the distinction is actually useful — a missing
file is recoverable (re-inspect and retry), a permission error generally
isn't (no tool-level workaround exists). Anything not more specifically
classified still gets a sane default rather than an empty/unknown
classification. `Tool.execute()` (the single dispatch point every tool
call goes through) catches *any* exception a tool implementation raises,
including ones that aren't `ToolError` at all, and converts it into a
failed `ToolResult` tagged `error_type="InternalError"` — no raw exception
from inside a tool can ever escape into the agent loop.

**A real bug this phase found and fixed**: `apply_change()` (file writes)
and `execute_command()` (terminal) were already fully guarded against
their own subprocess/OS errors, but `apply_git_operation()` — the function
that actually runs `git branch`/`git add`/`git commit`, called directly
from the agent loop with no `Tool.execute()` safety net around it — was
not. A `git` binary disappearing mid-session, or a hung `git` process,
would have raised an uncaught exception straight out of the agent loop's
generator, aborting the whole turn. It's fixed and covered by regression
tests (`TestGitUnavailableAndTimeout` in `tests/test_tools_git.py`) that
fail if the guard is ever removed.

### Retry policy

Only genuinely transient Ollama failures are retried: a connection error
or a request timeout gets up to `MAX_OLLAMA_RETRIES` (2) additional
attempts with a short backoff (1s, then 2s), yielding a `{"type": "retry",
attempt, max_attempts, wait_seconds}` event the UI can show as "Retrying
(2/3)...". A model-not-found error or a malformed-request-shaped API error
is **not** retried — an identical request would just fail identically, so
these fail immediately with a clear message instead. Tool-level failures
are never blindly retried either: a validation error is left for the model
to correct with different arguments, not silently re-attempted; a
permission-denied error doesn't get retried at all (see "Repetition and
failed-approach detection" below for what *does* happen if the model keeps
trying anyway).

### Loop and repetition protection

`MAX_TOOL_ITERATIONS` (10) was already Phase 2's hard ceiling on tool
calls within a single turn — audited and confirmed intact. New in Phase 8:
if the exact same tool name + arguments is requested
`MAX_CONSECUTIVE_IDENTICAL_CALLS` (3) times in a row — whether it keeps
"succeeding" with no progress (e.g. re-reading the same file over and
over) or keeps failing the same way (e.g. re-proposing a rejected edit) —
the agent loop stops actually executing/proposing it on the 3rd repeat and
instead tells the model directly: *"This exact call has been repeated N
times in a row without a different outcome. Stop repeating it..."*
(`{"type": "repetition_detected"}` event). This one mechanism covers both
"repetition detection" and "failed-approach detection" from the same
signature-tracking logic, reset the moment a genuinely different call
breaks the streak.

### Task recovery

A recoverable tool failure does not reset or discard the task. `TaskState`
(goal, plan, files inspected/modified, recent commands, errors) lives
independently of any single tool call's outcome; a failed step stays
visible with its status (the system prompt already instructs the model to
mark a step `blocked`/`failed` with a reason rather than claim success),
and the conversation continues in the same turn so the model can try a
different approach immediately — see the "Tool failure -> alternative
approach -> continue" integration test
(`tests/test_integration_workflows.py`) for this exact sequence exercised
end to end against real tools.

### Cancellation and subprocess cleanup

`execute_command()` (`agent/tools/terminal.py`) was rewritten from
`subprocess.run()` to `subprocess.Popen` with `start_new_session=True`,
polling in short increments instead of blocking for the full timeout in
one call. This means both the existing hard timeout *and* the new
`cancel_event`-driven Stop Task now terminate the **whole process group**
(`os.killpg`, SIGTERM then SIGKILL after a grace period) — not just the
direct child, and not just "give up waiting for it." A dedicated test
class actually runs a real `sleep 30` process and confirms the OS process
is genuinely dead afterward (`TestRealSubprocessTermination` in
`tests/test_tools_terminal.py`), not just that Python stopped waiting for
it. One known, deliberate limitation: an in-flight command's own OS
process is not killed *faster* than its normal termination path by
anything beyond this — there is no more aggressive mechanism, and none is
planned, since that's exactly the boundary that keeps command execution
predictable.

### Ollama failure handling

`OllamaClient._chat_updates()` now treats a JSON line that parses but
isn't the expected object shape (a bare list, string, or a `message` field
that isn't an object) as one malformed line to skip, not a crash —
`isinstance` checks guard every field before it's read. Server-unavailable,
timeout, model-not-found, and stream-interruption were already handled
pre-Phase-8 (Phase 1–2) and remain covered; Phase 8 adds the retry policy
above on top for the two transient cases.

### Logging and debugging

`agent/logging_config.py` wraps the stdlib `logging` module — no new
dependency. Quiet by default (WARNING and above, to stderr, so it never
interleaves with the Rich-rendered conversation on stdout); pass
`code-agent --debug` (works with or without `serve`) or set
`CODE_AGENT_DEBUG=1` for DEBUG-level detail, including every tool
failure's classification, every retry attempt, every repetition
interception, and (in server mode) a short random request ID per HTTP
request for correlating a client-visible error with the matching server
log line. Every log line is passed through a redaction filter that
recognizes common secret shapes (`Bearer <token>`, `token=`/`password=`/
`api_key=`-style key-value pairs) before being written — not a substitute
for the actual guarantee (nothing in this codebase logs raw request
bodies, environment dumps, or full file contents to begin with), just
defense in depth.

### Health checks and task/request IDs

`GET /health` (unchanged endpoint, additive fields) now optionally accepts
`?workspace_root=...` and reports `"workspace": "ok" | "not_found"`
alongside the existing agent/Ollama/model status — useful for telling
"agent down" apart from "agent up, but this specific folder went away."
Every `TaskState` carries a short `task_id` (visible in `/task/status` and
in log lines), regenerated on `/new`/`/task/new`, for correlating a
specific task's activity across a session.

### VS Code connection recovery

Verified, not just assumed: `AgentClient` (`vscode-extension/src/client.ts`)
never caches the server's host/port/token across calls — every request
re-reads `~/.code-agent/server.json` fresh. If `code-agent serve` is
restarted (new port, new token), the *very next* request from the same
long-lived `AgentClient` instance picks up the change automatically, no
VS Code window reload required. Two dedicated tests
(`vscode-extension/test/client.test.ts`) start a fake server, make a
request, kill it, start a second fake server with a different port and
token, and confirm the same client instance transparently reconnects —
and, separately, that a stale token from the old server is correctly
rejected by the new one (proving the token is actually being re-read, not
silently reused).

### Testing

New for Phase 8: `tests/test_logging_config.py` (redaction correctness and
the debug/quiet toggle), expanded failure-path coverage across
`tests/test_tools_filesystem.py`, `tests/test_tools_git.py`,
`tests/test_tools_terminal.py`, `tests/test_ollama_client.py`, and
`tests/test_loop.py` (retry policy, repetition detection, malformed
responses, permission errors, process-group termination — including real
subprocess tests, not just mocks), `tests/test_integration_workflows.py`
(six full end-to-end workflows: successful plan-driven task, test failure
→ fix → pass, tool failure → alternative approach → success, user
rejection → continue, Ollama failure → recovery, cancellation → state
preserved), and `tests/test_stress.py` (many tool calls in one turn, large
files/diffs, very long command output, repeated interruption cycles, and
bounded context growth across many turns). All of Phases 1–7's existing
tests continue passing unmodified in intent — a small number were updated
where Phase 8 deliberately changed behavior they asserted against (e.g. a
connection error now retries instead of failing immediately), with the
reason documented at the point of change.

### Qwen 3B benchmark

`agent/benchmark.py` is a small, repeatable local benchmark harness — no
external telemetry, nothing leaves the machine. It drives the exact same
`run_agent_turn()` the CLI and HTTP server use against a real local Ollama
server and a fresh temporary Git-repo fixture project, auto-approving
every confirm* prompt (the one deliberate difference from real usage,
since there's no human in the loop for an unattended run). Six tasks:
explain repository structure, find a known bug, modify a function, add a
test, run tests and fix a failure, and a multi-step change (fix the bug
*and* run tests). For each task it records success/failure (checked
against real file/Git/test-exit-code state, not just "did it produce
text"), wall-clock duration, tool-call count, retry count, and prompt/
completion token counts when Ollama reports them. Run it with:

```bash
python -m agent.benchmark --output benchmark_report.json
```

This is a baseline measurement for the 3B model, not an optimization
pass — Phase 8 deliberately does not tune the model or its prompting based
on these results; that's explicitly Phase 9's job.

## Security

The agent still cannot: delete files; read or write files outside the
project root; or read/create/edit key/credential-shaped files
(`*.pem`, `*.key`, `id_rsa*`, `credentials*.json`, `secrets.*`, `*.sqlite3`)
even inside the root. `.env`/`.env.*` is a deliberate, later exception to
that list (by request) — the agent can create/edit it directly like any
other file, but staging one into Git still requires an explicit
warning-and-override (see "Git integration"), independently of the edit
permission. It cannot write *any* file, run *any* command, or
change *any* Git state, anywhere, without a human explicitly approving it
first — the model only ever proposes. Approved commands are restricted to a
small allowlist with no shell, no network access, no installers, and no
background processes (see "Safe command execution"), and Git is restricted
to four narrow, validated operations (branch/stage/commit/push) with no
generic command tool at all and no way to reach a destructive one — push
included: it's a fixed, non-force `git push` of only the currently
checked-out branch to an already-configured remote (see "Git
integration").

None of this depends on the model behaving — it's enforced in Python.
Every tool call's arguments are validated (Pydantic, plus an explicit policy
layer for commands and Git operations), every path is resolved through
`ProjectRoot.resolve()` (`agent/project.py`), which rejects `..` traversal,
absolute paths outside the root, and symlinks that escape it, and every
write, command execution, or Git operation goes through its own propose →
show user → approve → act pipeline with no shortcut back to the model. This
was verified live, not just in tests: a real local `qwen2.5-coder:7b`
genuinely attempted to edit `.env` and to write outside the project root
during Phase 3 manual testing (`.env` was blocked outright at the time;
it's since been deliberately allowed for direct editing, see "Safe file
editing" above — the path-traversal rejection this test also exercised is
unaffected); `qwen2.5-coder:3b` (the default since Phase 4)
genuinely attempted `sudo reboot`, `curl example.com`, and a `run_command`
call with a literal shell-injection string as an argument, and — new in
Phase 5 — genuinely attempted `git push --force` via `run_command` and had
no tool available at all to even attempt `git reset --hard`. Every one of
these was rejected (or simply had no mechanism to run) before any diff or
approval prompt ever appeared. (`run_command` still denylists `git`
entirely today, unchanged; the later-added `git_push` tool is a completely
separate, narrow path that structurally cannot pass `--force` — see "Git
integration" above.)

**Planning changes none of this.** `create_plan` is the one tool in the app
where a bare Enter approves rather than rejects, but a plan is inert data —
`agent/planner.py`'s `Plan`/`PlanStep` carry no path, no command, no Git
ref, nothing capable of touching the filesystem, a process, or a repository.
Adopting a plan cannot skip or pre-approve any of the real actions above;
each one it leads to still pauses for its own separate, default-*reject*
approval. `update_plan`/`get_plan` only ever read or write step statuses on
an already-approved plan held in memory.

## What is implemented now (Phase 8)

Everything from Phases 1–7 (below), plus reliability and failure recovery
— see "Reliability and Failure Recovery" above for the full picture. No
new user-facing capability was added; every item below makes an existing
capability fail safely instead of crashing or corrupting state:

- `error_type`/`recoverable` classification on every `ToolResult`, backed
  by a small `ToolError` subclass taxonomy (`ValidationFailedError`,
  `NotFoundError`, `PermissionDeniedError`, `StaleStateError`,
  `ToolTimeoutError`, `ExternalToolUnavailableError`)
- A real bug fix: `apply_git_operation()` (called directly from the agent
  loop, unguarded) could previously crash a whole turn if `git` was
  unavailable or hung; it's now self-guarded like `apply_change()` and
  `execute_command()` already were
- Bounded retries (`MAX_OLLAMA_RETRIES`, short backoff) for transient
  Ollama connection/timeout errors specifically — not for non-transient
  ones, and not for tool-level failures
- Repetition/failed-approach detection: the same tool call repeated
  `MAX_CONSECUTIVE_IDENTICAL_CALLS` times in a row is intercepted and the
  model is told to change approach, instead of burning the whole
  iteration budget or repeatedly re-prompting for the same approval
- `execute_command()` rewritten to `Popen` + polling, so both the existing
  timeout and the new `cancel_event`-driven Stop Task genuinely terminate
  the whole process group (not just the direct child, not just "stop
  waiting")
- `agent/logging_config.py`: quiet-by-default structured logging with
  secret redaction, enabled via `code-agent --debug` /
  `CODE_AGENT_DEBUG=1`
- `task_id` on `TaskState`, short random `request_id`s on HTTP requests,
  for correlating logs
- `GET /health?workspace_root=...` reports workspace validity alongside
  agent/Ollama/model status
- Verified (not just assumed): the VS Code extension automatically
  reconnects after `code-agent serve` restarts, with no window reload
- `agent/benchmark.py`: a local, repeatable Qwen 3B benchmark harness
- Extensive new tests: unit tests for every failure path above, 6
  end-to-end integration-test workflows, and a stress/memory test suite
  — see "Testing" above for the full list; all pre-existing tests
  continue passing

## What is implemented now (Phase 7)

Everything from Phases 1–6 (below), plus VS Code integration:

- `agent/server.py`: a stdlib `ThreadingHTTPServer`-based local agent
  server (`code-agent serve`) exposing `/health`, `/chat`, `/chat/confirm`,
  `/task/status`, `/task/stop`, `/task/new` — see "VS Code Integration"
  above for the full protocol, security posture, and workspace isolation
- Additive `cancel_event` support threaded through `OllamaClient` and
  `run_agent_turn()` for Stop Task, plus a new `OllamaCancelledError` and a
  `{"type": "cancelled"}` event — `None` by default, so the CLI's own calls
  are completely unaffected
- `code-agent serve` subcommand dispatch in `cli.py`'s `main()`; the
  no-argument interactive CLI path is untouched
- `vscode-extension/`: a separate, unpublished TypeScript VS Code
  extension (Webview chat panel, native diff integration, selection-based
  commands, Stop/New Task, a Connected/Offline + model status indicator)
  that contains no agent logic of its own — see "VS Code Integration"
- 26 new Python tests (`tests/test_server.py`) covering binding to
  127.0.0.1 only, token auth (missing/invalid/valid), the health/chat/
  task-status/task-stop/task-new endpoints, workspace isolation, malformed
  requests, and a live confirm → `/chat/confirm` → applied round trip;
  9 new `cancel_event` unit tests across `test_ollama_client.py` and
  `test_loop.py`; 14 TypeScript tests (`vscode-extension/test/`, run via
  `npm test` / Node's built-in `node:test`) covering NDJSON line-splitting,
  token-file parsing, and `AgentClient` against a throwaway local server
- Live-verified end-to-end against a real local Ollama server and
  `qwen2.5-coder:3b` speaking to `code-agent serve` over HTTP: chat about
  project structure, an Explain-Selection-style request, a full edit
  propose → `confirm` event → separate `/chat/confirm` approval →
  `change_applied` → file actually modified on disk, a `run_command`
  approved and (separately) rejected with correct enforcement both ways, a
  Git stage → commit approval chain producing a real commit, Stop Task
  cutting off a turn mid-flight, New Task resetting `/task/status` to
  empty, and path-traversal/sensitive-file protection both still rejecting
  the model exactly as they do in the CLI

## What is implemented (Phase 6)

Everything from Phases 1–5, plus:

- `create_plan`, `update_plan`, `get_plan` — see "Planning and task state"
- `agent/planner.py`: plain-data `Plan`/`PlanStep` with a status
  (`pending`/`in_progress`/`completed`/`blocked`/`failed`) per step, and
  `agent/task_state.py`'s `TaskState`: the current task's goal, plan, files
  inspected/modified, recent commands (with parsed outcomes, not raw
  output), errors, and Git actions — process-lifetime only, nothing persisted
- `agent/context_manager.py`: turns tool results into those compact
  `TaskState` entries, regenerates a bounded "task memory" section of the
  system prompt every round-trip instead of letting it accumulate, and
  compacts the raw conversation sent to Ollama — superseding an old
  duplicate `read_file` result, invalidating one that predates a later edit
  of the same file, and (only past a size budget) trimming old tool output,
  all without ever touching the system prompt or the most recent messages
- A fourth confirmation kind, `confirm_plan`, on the same `send()`-based
  pause used for file/command/Git approvals — but the only one that
  defaults to *approve* on a bare Enter, since a plan has no side effects
- `/new`: resets the plan, task memory, file-read cache, and conversation
  in one command, without touching project files or Git history
- Ctrl+C now preserves and displays whatever plan progress exists so far
  ("Task paused...") instead of just discarding the turn silently
- An updated system prompt: create a plan only for genuinely multi-step
  work, never for a small single-action request; mark a step `completed`
  only once it's actually verified, not just edited; use `blocked`/`failed`
  with a reason instead of claiming success; a plan is never permission to
  skip a file/command/Git approval
- Tests for the plan data structure, `TaskState`'s tracking and
  cache-invalidation-on-modify, the context compaction logic (dedup,
  staleness, size-based trimming, and that user/system messages are never
  touched), the planning tools' validation, the full propose → confirm →
  adopt/reject loop for plans (including that `update_plan` after approval
  correctly mutates the shared task state), and the plan approval prompt's
  default-yes parsing specifically
- Live-verified end-to-end against a real local Ollama server and
  `qwen2.5-coder:3b`: a genuine `create_plan` call rendered and approved
  with a bare `y`, a full `update_plan` progression through
  `in_progress`→`completed` for one step and into the next with the
  progress view re-rendering live after each call, a rejected `run_command`
  correctly reported back without any false success claim, and `/new`
  correctly resetting state with its confirmation message

## What is NOT implemented (out of scope)

- Deleting files (`delete_file`)
- Any GitHub/GitLab/remote-repository *integration* (PRs, issues, CI,
  hosted API calls), cloud execution, or remote agents. (`git_push` — a
  single, narrow, non-force push of the current branch to an
  already-configured remote — was added later by explicit request; see
  "Git integration" above. That's the full extent of remote Git support:
  no PRs, no GitHub API, no remote add/fetch/pull.)
- Multi-agent coordination, sub-agents, or any additional model processes;
  autonomous background agents
- Persistent/long-term memory across sessions or processes (task state is
  per-workspace, in-memory, and cleared on `/new`/`/task/new` or process
  exit — no vector database, no embeddings)
- A standalone web application or mobile application (VS Code's native
  Webview is the only UI surface added in Phase 7)
- Any additional CLI/chat commands beyond `exit` / `quit` / `/new`
- Batched upfront review of multiple proposed changes/commands (each is
  currently reviewed as it's proposed, not pre-planned as a numbered set)
- Installing dependencies automatically, running arbitrary scripts beyond the
  allowlisted shapes, background/long-running processes (e.g. dev servers)
- Force-killing an in-flight subprocess command specifically via Stop
  Task — it still relies on Phase 4's existing hard per-command timeout
- Full multi-root VS Code workspace support — a multi-root window uses its
  first folder for the general chat panel (a warning is shown), though
  selection-based commands (Explain/Fix/Ask) correctly use whichever
  folder the active file actually belongs to

## Running tests

```bash
pip install -r requirements.txt
pytest
```

All Ollama calls and tool execution are mocked/use temp directories, so the
full suite runs without Ollama installed or running. This includes
`agent/server.py`'s tests, which start a real local HTTP server on an
OS-assigned port and drive it with real requests — only `OllamaClient` is
mocked. `tests/test_tools_terminal.py` additionally runs a handful of
tests against a genuine short-lived OS process (`sleep`) to prove
timeout/cancellation actually terminate it, not just stop waiting for it.

To run the VS Code extension's tests (no VS Code host required):

```bash
cd vscode-extension
npm install
npm test
```

To run the Qwen 3B benchmark (requires a running local Ollama server with
the model pulled — this is a baseline measurement, not part of the
pytest suite, and takes a few minutes):

```bash
python -m agent.benchmark --output benchmark_report.json
```

## Project structure

```
code-agent/
├── agent/
│   ├── __init__.py
│   ├── cli.py               # terminal UI: banner, input loop, event rendering,
│   │                         # diff/plan display, file/command/Git/plan approvals,
│   │                         # `serve` subcommand dispatch, --debug flag
│   ├── server.py             # local HTTP agent server for the VS Code extension
│   │                         # (code-agent serve): auth, session isolation, NDJSON
│   ├── loop.py               # the tool-calling agent loop (incl. confirm/apply
│   │                         # for file changes, commands, Git ops, and plans;
│   │                         # cancel_event, Ollama retry policy, repetition
│   │                         # detection)
│   ├── logging_config.py     # structured logging + secret redaction (Phase 8)
│   ├── benchmark.py           # local Qwen 3B benchmark harness (Phase 8)
│   ├── diff.py                # ProposedChange + unified diff generation
│   ├── command_policy.py      # run_command's allowlist-based validator
│   ├── git_policy.py          # branch/staging-path/commit-message validator
│   │                         # + ProposedGitOperation
│   ├── planner.py             # Plan / PlanStep: plain-data plan representation
│   ├── task_state.py          # TaskState: current-task-only short-term memory
│   │                         # + task_id (Phase 8)
│   ├── context_manager.py     # tool results -> TaskState, system-prompt memory
│   │                         # injection, conversation compaction
│   ├── ollama_client.py      # Ollama /api/chat communication (streaming + tools;
│   │                         # cancel_event, malformed-response hardening)
│   ├── project.py            # ProjectRoot: path resolution & traversal protection
│   └── tools/
│       ├── __init__.py       # builds the default tool registry
│       ├── base.py           # Tool / ToolResult / ToolError + error taxonomy
│       │                     # (NotFoundError, PermissionDeniedError, etc.)
│       ├── registry.py       # ToolRegistry: schemas + dispatch
│       ├── filesystem.py     # list_files, read_file
│       ├── search.py         # search_files (ripgrep + fallback)
│       ├── editing.py         # edit_file, write_file, apply_change (atomic write)
│       ├── state.py           # FileStateTracker for stale-edit detection
│       ├── terminal.py        # run_command tool + execute_command (Popen,
│       │                     # process-group termination, cancel_event)
│       ├── git.py             # git_status/diff/log + git_create_branch/stage/
│       │                     # commit + apply_git_operation (self-guarded)
│       └── planning.py        # create_plan, update_plan, get_plan
├── tests/
│   ├── test_ollama_client.py
│   ├── test_project.py
│   ├── test_tools_filesystem.py
│   ├── test_tools_search.py
│   ├── test_tools_editing.py
│   ├── test_tools_terminal.py
│   ├── test_tools_git.py
│   ├── test_tools_planning.py
│   ├── test_command_policy.py
│   ├── test_git_policy.py
│   ├── test_planner.py
│   ├── test_task_state.py
│   ├── test_context_manager.py
│   ├── test_diff.py
│   ├── test_cli_approval.py
│   ├── test_loop.py
│   ├── test_server.py        # local HTTP agent server tests
│   ├── test_logging_config.py    # redaction + debug toggle (Phase 8)
│   ├── test_integration_workflows.py  # 6 end-to-end workflows (Phase 8)
│   └── test_stress.py            # stress + memory/resource tests (Phase 8)
├── vscode-extension/          # separate TypeScript project, not published to
│   │                         # the Marketplace -- install locally as a .vsix
│   ├── src/
│   │   ├── extension.ts      # activation, command registration
│   │   ├── client.ts         # HTTP/NDJSON client (no `vscode` dependency)
│   │   ├── panel.ts          # Webview chat panel
│   │   ├── commands.ts       # command handlers (workspace/selection resolution)
│   │   └── diffProvider.ts   # native VS Code diff editor integration
│   ├── media/                # webview HTML/CSS/JS (main.css, main.js)
│   ├── test/                 # node:test unit tests for client.ts
│   ├── package.json
│   ├── tsconfig.json
│   └── README.md             # extension install/usage/troubleshooting
├── requirements.txt
├── pyproject.toml
├── .gitignore
└── README.md
```
