"""Interactive terminal chat loop for the local coding agent.

This module owns terminal I/O and conversation state. All communication
with Ollama goes through OllamaClient, and all project inspection/editing
goes through the tool registry — this file just renders what happens and
collects the user's approval for proposed file changes.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from rich.console import Console
from rich.markup import escape
from rich.panel import Panel

from . import context_manager
from .command_policy import ApprovedCommand
from .diff import ProposedChange, unified_diff_text
from .git_policy import ProposedGitOperation
from .logging_config import configure_logging, debug_enabled_from_env, get_logger
from .loop import MAX_TOOL_ITERATIONS, run_agent_turn
from .ollama_client import DEFAULT_HOST, DEFAULT_MODEL, OllamaClient
from .planner import Plan
from .project import ProjectRoot
from .task_state import TaskState
from .tools import FileStateTracker, build_default_registry
from .tools.terminal import CommandExecutionResult, describe_command

SYSTEM_PROMPT_TEMPLATE = """You are a coding assistant operating inside a specific project directory:

{project_root}

You have read-only inspection tools (list_files, read_file, search_files, git_status, git_diff, \
git_log), controlled editing tools (edit_file, write_file), one controlled command-execution tool \
(run_command), and controlled Git modification tools (git_create_branch, git_stage, git_commit). \
Use inspection tools only for questions that actually require looking at this project's files, \
structure, or Git history — never guess about file names, code, or behavior you have not actually \
inspected.

For general programming questions that have nothing to do with this specific project (language \
features, concepts, debugging advice, writing a standalone example, etc.), answer directly from \
your own knowledge — do not call a tool and do not claim you're unable to help just because a \
question isn't about this project.

Before editing anything:
1. Inspect the relevant file(s) with read_file — never propose an edit against code you haven't \
actually read in this conversation.
2. Understand the existing implementation before changing it.
3. Make the smallest reasonable change that accomplishes the request.
4. Use edit_file for existing files (old_text must match the file's real content exactly once —
include enough surrounding context to make the match unambiguous). Use write_file only for files \
that don't exist yet; it fails if the file is already there.

Every edit_file and write_file call only proposes a change — it is never applied automatically. \
The user is shown a diff and must explicitly approve it before anything is written to disk. \
Because of this:
- Never claim a file was modified, created, or fixed until the tool result confirms the user \
approved it and the write succeeded. A pending or rejected proposal is not a completed change.
- If the user rejects a change, do not immediately re-propose the same edit — acknowledge the \
rejection and ask what they'd like instead, or wait for further instructions.
- If a tool reports the target text wasn't found, appeared more than once, or the file changed \
since it was last read, don't guess — read_file again and propose a more precise edit.

run_command runs one local development/test command at a time (e.g. pytest, ruff check ., mypy ., \
npm test, npm run <script>, python -m pytest, or python manage.py test) and, like editing, only \
proposes it — the user is shown exactly what would run and must approve it before it executes. \
Only a small allowlist of programs and argument shapes is permitted; anything else (shell syntax, \
network tools, installers, background processes, arbitrary scripts) is rejected automatically \
before the user is ever asked, so don't waste a turn retrying a rejected command with different \
shell tricks — it will be rejected the same way. Decide what to run from real project evidence, \
not guesswork: check for pytest.ini/pyproject.toml/a tests/ directory before proposing pytest; \
read package.json's "scripts" before proposing an npm command; check for manage.py before proposing \
`python manage.py test`. This agent never installs dependencies (pip install, npm install, etc.) — \
if a command fails because something isn't installed, tell the user what's missing and let them \
install it themselves. If a proposed command is rejected or fails, read the result and adjust \
rather than repeating the same command unchanged.

You are working inside a Git repository when Git is available — check with git_status if you're \
not sure. You can inspect Git state freely and without approval using git_status, git_diff, and \
git_log; none of them change anything. You can request controlled Git modifications using \
git_create_branch, git_stage, git_commit, and git_push — like editing and running commands, each \
of these only proposes the operation, and the user is shown exactly what would happen and must \
approve it before anything changes. Before proposing a commit, inspect the staged diff with \
git_diff(staged=true) so you actually know what you're committing, and stage files with git_stage \
first if nothing (or the wrong thing) is staged yet. If the user asks you to fix something and \
commit it, still show the diff, the files to stage, and the proposed commit message, and get \
approval at each step — never skip straight to committing because the user said to. git_push only \
ever pushes the branch that's currently checked out to an already-configured remote (default \
'origin') with a plain, non-force push — it cannot check out a different branch first, cannot add \
a new remote, and cannot force-push; if a push is rejected (e.g. the remote has commits you don't \
have locally), report that plainly rather than trying to work around it.

Never claim a branch was created, files were staged, a commit was made, or a push succeeded until \
the tool result confirms the user approved it and it succeeded. Never use Git to discard the \
user's existing work: this agent has no reset, clean, checkout/restore-to-discard, rebase, merge, \
force-push, or branch -D capability, and you must not try to improvise one — if the working tree \
has unrelated modifications, leave them alone and only stage/commit what the user actually asked \
for. If git_status shows changes you didn't expect, inspect them with git_diff before doing \
anything else instead of assuming they're safe to override.

Neither editing, running commands, nor Git modifications (including pushing) happen automatically \
— in every case, never claim an action completed until the tool result confirms the user approved \
it and it succeeded. A pending or rejected proposal is not a completed action. Only mention a \
limitation if the user actually asks for something it covers; do not bring it up otherwise.

For multi-step tasks that genuinely span several files or concerns (e.g. "add JWT authentication", \
"add password reset functionality"), call create_plan first with a short goal and a handful of \
concrete steps, before making any changes — the user is shown the plan and asked to proceed. For \
small, single-action requests (e.g. "rename this variable", "change this button text"), skip \
create_plan and just do it directly. Once a plan is approved, mark a step in_progress before \
working on it and completed only once it is genuinely done — verified (e.g. tests passing), not \
just edited — using update_plan; use blocked or failed with a short note if you cannot continue a \
step rather than claiming success. A plan is not permission to skip approvals: every file edit, \
command, and Git operation still needs its own separate approval exactly as before.

Never claim to have inspected a file or directory you have not actually called a tool on. If a \
tool call fails or turns up nothing useful, say so instead of making something up."""

EXIT_COMMANDS = {"exit", "quit"}
MAX_DIFF_DISPLAY_LINES = 200


def build_banner(model: str, project_root: Path) -> Panel:
    text = (
        f"[bold]Local Coding Agent[/bold]\n"
        f"[dim]Model: {escape(model)}[/dim]\n"
        f"[dim]Project: {escape(str(project_root))}[/dim]"
    )
    return Panel(text, expand=False, border_style="cyan")


def render_diff(console: Console, change: ProposedChange) -> None:
    """Print a colored unified diff for a proposed change. All diff content
    is arbitrary code from the model/filesystem, so it's always escaped —
    never trust it to be safe as Rich markup (see the Phase 2 bracket bug).
    """
    diff_text = unified_diff_text(change)
    label = "New file" if change.kind == "create" else "Modified file"
    rule = "─" * min(console.width, 60)

    console.print(f"\n[bold]{escape(label)}:[/bold] {escape(change.path)}")
    console.print(rule)

    lines = diff_text.splitlines()
    for line in lines[:MAX_DIFF_DISPLAY_LINES]:
        safe = escape(line)
        if line.startswith("+++") or line.startswith("---"):
            console.print(f"[dim]{safe}[/dim]")
        elif line.startswith("@@"):
            console.print(f"[cyan]{safe}[/cyan]")
        elif line.startswith("+"):
            console.print(f"[green]{safe}[/green]")
        elif line.startswith("-"):
            console.print(f"[red]{safe}[/red]")
        else:
            console.print(safe)

    if len(lines) > MAX_DIFF_DISPLAY_LINES:
        omitted = len(lines) - MAX_DIFF_DISPLAY_LINES
        console.print(f"[dim]... {omitted} more line(s) omitted ...[/dim]")

    console.print(rule)


def _ask_approval(console: Console) -> str:
    """Prompt for approval. Returns "yes", "all", or "no" — anything other
    than an explicit y/yes/a defaults to "no", including a bare Enter.
    """
    raw = console.input(f"Apply this change? {escape('[y/N/a]')}: ").strip().lower()
    if raw in ("y", "yes"):
        return "yes"
    if raw == "a":
        return "all"
    return "no"


def _handle_confirm(console: Console, change: ProposedChange, turn_state: dict) -> bool:
    render_diff(console, change)
    if turn_state["approve_all"]:
        console.print("[dim](auto-approved — \"approve all\" is active for this turn)[/dim]")
        return True

    action = _ask_approval(console)
    if action == "all":
        turn_state["approve_all"] = True
        return True
    return action == "yes"


MAX_COMMAND_OUTPUT_DISPLAY_LINES = 60


def _ask_command_approval(console: Console) -> bool:
    """Only an explicit y/yes runs the command; anything else, including a
    bare Enter, rejects it. No "approve all" shortcut here — unlike file
    edits, every proposed command is confirmed individually.
    """
    raw = console.input(f"Allow this command? {escape('[y/N]')}: ").strip().lower()
    return raw in ("y", "yes")


def _handle_command_confirm(console: Console, cmd: ApprovedCommand) -> bool:
    console.print("\n[bold]Agent wants to run:[/bold]\n")
    console.print(f"  {escape(describe_command(cmd))}")
    console.print("\n[bold]Working directory:[/bold]")
    console.print(f"  {escape(str(cmd.cwd))}\n")
    return _ask_command_approval(console)


def render_command_result(console: Console, result: CommandExecutionResult) -> None:
    """Print a capped summary for the human. The model receives the fuller
    (still truncated, see terminal.py) text separately via the tool result.
    """
    if result.cancelled:
        console.print("[yellow]Command was cancelled and terminated.[/yellow]")
    elif result.timed_out:
        console.print("[yellow]Command timed out and was terminated.[/yellow]")
    elif result.exit_code == 0:
        console.print("[green]Command finished (exit code 0).[/green]")
    else:
        console.print(f"[yellow]Command finished (exit code {result.exit_code}).[/yellow]")

    for stream_name, text in (("stdout", result.stdout), ("stderr", result.stderr)):
        if not text:
            continue
        lines = text.splitlines()
        console.print(f"[dim]--- {stream_name} ---[/dim]")
        for line in lines[:MAX_COMMAND_OUTPUT_DISPLAY_LINES]:
            console.print(line, markup=False, highlight=False)
        if len(lines) > MAX_COMMAND_OUTPUT_DISPLAY_LINES:
            omitted = len(lines) - MAX_COMMAND_OUTPUT_DISPLAY_LINES
            console.print(f"[dim]... {omitted} more line(s) omitted (the full output was sent to the model) ...[/dim]")
    console.print()


def _ask_git_approval(console: Console, prompt: str) -> bool:
    """Only an explicit y/yes proceeds; anything else, including a bare
    Enter, rejects. No "approve all" shortcut here either — every proposed
    Git operation is confirmed individually. `prompt` must already be
    markup-escaped by the caller.
    """
    raw = console.input(f"{prompt} ").strip().lower()
    return raw in ("y", "yes")


def _handle_git_confirm(console: Console, op: ProposedGitOperation) -> bool:
    if op.kind == "create_branch":
        console.print("\n[bold]Agent wants to create branch:[/bold]\n")
        console.print(f"  {escape(op.branch_name)}\n")
        return _ask_git_approval(console, escape("Create this branch? [y/N]:"))

    if op.kind == "stage":
        console.print("\n[bold]Agent wants to stage:[/bold]\n")
        for path in op.paths:
            marker = " [red](sensitive)[/red]" if path in op.sensitive_paths else ""
            console.print(f"  {escape(path)}{marker}")
        if op.sensitive_paths:
            console.print()
            for path in op.sensitive_paths:
                console.print(
                    f"[bold red]WARNING:[/bold red] {escape(path)} matches a sensitive-file "
                    "pattern and may contain secrets."
                )
            console.print()
            return _ask_git_approval(console, escape("Are you sure you want to stage it? [y/N]:"))
        console.print()
        return _ask_git_approval(console, escape("Stage these files? [y/N]:"))

    if op.kind == "commit":
        console.print("\n[bold]Agent proposes commit:[/bold]\n")
        console.print("[bold]Message:[/bold]")
        console.print(f"  {escape(op.message)}\n")
        console.print("[bold]Files currently staged:[/bold]\n")
        for path in op.expected_staged_files:
            console.print(f"  {escape(path)}")
        console.print()
        return _ask_git_approval(console, escape("Create this commit? [y/N]:"))

    if op.kind == "push":
        console.print("\n[bold]Agent wants to push:[/bold]\n")
        destination = f"{op.remote} ({op.remote_url})" if op.remote_url else op.remote
        console.print(f"  Branch:      {escape(op.branch_name)}")
        console.print(f"  Remote:      {escape(destination)}\n")
        console.print("[bold]Commits to be pushed:[/bold]")
        for line in (op.commits_preview or "").splitlines():
            console.print(f"  {escape(line)}")
        console.print()
        return _ask_git_approval(console, escape("Push to the remote? [y/N]:"))

    return False  # pragma: no cover - unknown kind, safe default


def render_plan(console: Console, plan: Plan) -> None:
    console.print("\n[bold]I'll handle this in the following steps:[/bold]\n")
    for line in plan.render_lines():
        console.print(f"  {escape(line)}")
    console.print()


def render_plan_progress(console: Console, task_state: TaskState) -> None:
    if task_state.plan is None:
        return
    console.print("\n[bold]Plan:[/bold]\n")
    for line in task_state.plan.render_lines():
        console.print(f"  {escape(line)}")
    console.print()
    if task_state.plan.is_complete():
        console.print("[bold green]✓ All plan steps are complete.[/bold green]\n")
    elif task_state.plan.is_blocked():
        console.print("[bold yellow]⚠ Task blocked — see the step above for the reason.[/bold yellow]\n")


def _ask_plan_approval(console: Console) -> bool:
    """Unlike every other approval prompt in this app, a bare Enter here
    means yes. A plan has no side effects of its own to guard against —
    approving it is not permission to skip the separate, default-reject
    approval every actual file edit, command, or Git operation still needs.
    Only an explicit n/no rejects it.
    """
    raw = console.input(f"{escape('Proceed? [Y/n]:')} ").strip().lower()
    return raw not in ("n", "no")


def _handle_plan_confirm(console: Console, plan: Plan) -> bool:
    render_plan(console, plan)
    return _ask_plan_approval(console)


def render_turn(console: Console, client: OllamaClient, registry, messages: list, tracker, task_state) -> None:
    """Consume one agent turn's events and render them to the terminal.

    Each model round-trip is fully buffered before it's shown (see loop.py
    for why), so a "thinking" spinner runs for the duration of every wait —
    stopped only to print a discrete line (or run an approval prompt), then
    restarted. Proposed file changes pause the underlying generator via
    generator.send() until the user answers — see run_agent_turn's "confirm"
    event.
    """
    inspecting_announced = False
    prefix_printed = False
    turn_state = {"approve_all": False}

    gen = run_agent_turn(
        client, registry, messages, tracker=tracker, task_state=task_state, max_iterations=MAX_TOOL_ITERATIONS
    )

    with console.status("[dim]Agent is thinking...[/dim]", spinner="dots") as status:
        send_value = None
        while True:
            try:
                event = gen.send(send_value)
            except StopIteration:
                break
            send_value = None
            etype = event["type"]

            if etype == "tool_call":
                status.stop()
                if not inspecting_announced:
                    console.print("[dim]Agent is inspecting the project...[/dim]\n")
                    inspecting_announced = True
                console.print(f"  [dim]→[/dim] {escape(event['display'])}")
                if event["name"] == "update_plan" and task_state.plan is not None:
                    render_plan_progress(console, task_state)
                status.start()

            elif etype == "tool_error":
                console.print(f"    [red]! {escape(event['message'])}[/red]")

            elif etype == "repetition_detected":
                console.print(
                    f"    [yellow]! Repeated {escape(event['name'])} call "
                    f"{event['count']} times without progress -- asked the agent to change "
                    "approach.[/yellow]"
                )

            elif etype == "confirm":
                status.stop()
                if not inspecting_announced:
                    console.print("[dim]Agent is inspecting the project...[/dim]\n")
                    inspecting_announced = True
                console.print("\n[bold]Proposed change:[/bold]")
                send_value = _handle_confirm(console, event["change"], turn_state)
                status.start()

            elif etype == "change_applied":
                console.print(f"[green]✓ Change applied successfully.[/green] ({escape(event['path'])})\n")

            elif etype == "change_rejected":
                console.print("[yellow]✗ Change rejected. No files were modified.[/yellow]\n")

            elif etype == "confirm_command":
                status.stop()
                if not inspecting_announced:
                    console.print("[dim]Agent is inspecting the project...[/dim]\n")
                    inspecting_announced = True
                send_value = _handle_command_confirm(console, event["command"])
                status.start()

            elif etype == "command_started":
                console.print(f"\n[dim]Running {escape(event['display'])}...[/dim]\n")

            elif etype == "command_result":
                render_command_result(console, event["result"])

            elif etype == "command_rejected":
                console.print("[yellow]✗ Command was not executed.[/yellow]\n")

            elif etype == "confirm_git_operation":
                status.stop()
                if not inspecting_announced:
                    console.print("[dim]Agent is inspecting the project...[/dim]\n")
                    inspecting_announced = True
                send_value = _handle_git_confirm(console, event["operation"])
                status.start()

            elif etype == "git_operation_applied":
                console.print(f"[green]✓ {escape(event['message'])}[/green]\n")

            elif etype == "git_operation_rejected":
                console.print("[yellow]✗ Git operation not performed. Nothing was changed.[/yellow]\n")

            elif etype == "confirm_plan":
                status.stop()
                if not inspecting_announced:
                    console.print("[dim]Agent is inspecting the project...[/dim]\n")
                    inspecting_announced = True
                send_value = _handle_plan_confirm(console, event["plan"])
                status.start()

            elif etype == "plan_approved":
                console.print("[green]✓ Plan approved.[/green]")
                render_plan_progress(console, task_state)

            elif etype == "plan_rejected":
                console.print("[yellow]Plan not approved.[/yellow]\n")

            elif etype == "content":
                status.stop()
                if not prefix_printed:
                    if inspecting_announced:
                        console.print()
                    console.print("[bold cyan]Agent[/bold cyan] > ", end="")
                    prefix_printed = True
                console.print(event["text"], end="", highlight=False, soft_wrap=True, markup=False)

            elif etype == "final":
                status.stop()
                if not prefix_printed:
                    console.print("[bold cyan]Agent[/bold cyan] > ", end="")
                    console.print(event["text"], end="", highlight=False, soft_wrap=True, markup=False)
                console.print()
                console.print()

            elif etype == "retry":
                console.print(
                    f"[yellow]Connection to Ollama failed ({escape(event['reason'])}). "
                    f"Retrying ({event['attempt']}/{event['max_attempts'] - 1})...[/yellow]"
                )

            elif etype == "cancelled":
                status.stop()
                console.print("\n[yellow]Task stopped.[/yellow]")

            elif etype == "error":
                status.stop()
                console.print(f"\n[bold red]Ollama error:[/bold red] {escape(event['message'])}")

            elif etype == "max_iterations":
                status.stop()
                console.print(
                    f"\n[yellow]Stopped after {MAX_TOOL_ITERATIONS} tool calls without a "
                    "final answer. Try a more specific question.[/yellow]"
                )


NEW_TASK_COMMANDS = {"/new"}


def _fresh_session_state(project: ProjectRoot, project_root_path: Path):
    """Builds a brand-new tracker/task_state/registry/messages set -- used
    both for the initial session and for /new, so the two can never drift
    apart. Nothing here touches project files or Git history; it only
    resets in-memory state for this process.
    """
    tracker = FileStateTracker()
    task_state = TaskState()
    registry = build_default_registry(project, tracker, task_state)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_TEMPLATE.format(project_root=project_root_path)}
    ]
    return tracker, task_state, registry, messages


def _run_serve_command(argv: list, debug: bool) -> None:
    """Handle `code-agent serve` -- starts the local HTTP server that the
    VS Code extension talks to (see agent/server.py). Deferred import so
    the ordinary interactive CLI path never pays for importing the server
    module, and to avoid a circular import (server.py imports
    _fresh_session_state from this module). `debug` is decided once by
    main() (a bare --debug works before or after "serve") rather than
    parsed twice.
    """
    parser = argparse.ArgumentParser(prog="code-agent serve")
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port to bind on 127.0.0.1 (default 8765, falls back to an OS-assigned port if busy).",
    )
    args = parser.parse_args(argv)

    configure_logging(debug=debug or debug_enabled_from_env())

    from .server import DEFAULT_PORT, run_server

    run_server(port=args.port if args.port is not None else DEFAULT_PORT)


def main() -> None:
    debug = "--debug" in sys.argv
    if debug:
        sys.argv = [a for a in sys.argv if a != "--debug"]

    if len(sys.argv) > 1 and sys.argv[1] == "serve":
        _run_serve_command(sys.argv[2:], debug=debug)
        return

    configure_logging(debug=debug or debug_enabled_from_env())
    logger = get_logger("cli")
    logger.debug("Starting interactive CLI session.")

    console = Console()

    host = os.environ.get("OLLAMA_HOST", DEFAULT_HOST)
    model = os.environ.get("OLLAMA_MODEL", DEFAULT_MODEL)
    project_root_path = Path.cwd().resolve()

    client = OllamaClient(host=host, model=model)
    project = ProjectRoot(project_root_path)
    tracker, task_state, registry, messages = _fresh_session_state(project, project_root_path)

    console.print(build_banner(model, project_root_path))
    console.print()

    if not client.check_connection():
        console.print(
            "[bold red]Could not connect to Ollama.[/bold red] Make sure Ollama is "
            f"running at [cyan]{escape(host)}[/cyan] (start it with: [cyan]ollama serve[/cyan])."
        )
        sys.exit(1)

    while True:
        try:
            user_input = console.input("[bold green]You[/bold green] > ").strip()
        except KeyboardInterrupt:
            console.print("\n[dim]Goodbye![/dim]")
            break
        except EOFError:
            console.print("\n[dim]Goodbye![/dim]")
            break

        if not user_input:
            continue

        if user_input.lower() in EXIT_COMMANDS:
            console.print("[dim]Goodbye![/dim]")
            break

        if user_input.lower() in NEW_TASK_COMMANDS:
            tracker, task_state, registry, messages = _fresh_session_state(project, project_root_path)
            console.print("[dim]Started a new task. Project files and Git history are untouched.[/dim]\n")
            continue

        history_len_before = len(messages)
        messages.append({"role": "user", "content": user_input})

        try:
            render_turn(console, client, registry, messages, tracker, task_state)
        except KeyboardInterrupt:
            console.print("\n[dim](response interrupted)[/dim]")
            del messages[history_len_before:]
            if task_state.plan is not None:
                console.print("\n[bold]Task paused.[/bold]")
                render_plan_progress(console, task_state)
            continue
        except Exception as exc:  # pragma: no cover - last-resort safety net
            console.print(f"\n[bold red]Unexpected error:[/bold red] {escape(str(exc))}")
            del messages[history_len_before:]
            continue


if __name__ == "__main__":
    main()
