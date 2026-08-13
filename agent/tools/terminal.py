"""Approval-gated local command execution: run_command.

Mirrors the propose/apply split editing.py established in Phase 3.
run_command's run() only validates the request (via command_policy.py) and
builds an ApprovedCommand describing exactly what would run -- it never
executes anything itself. The agent loop only calls execute_command() after
the user has explicitly approved, and even then the subprocess is invoked
as an argv list (never shell=True), from the project root, with a hard
timeout, a stripped-down environment, and truncated output -- never the
model's own say-so, and never unrestricted shell access.
"""
from __future__ import annotations

import os
import signal
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import List, Optional

from pydantic import BaseModel, Field

from ..command_policy import (
    DEFAULT_TIMEOUT_SECONDS,
    MAX_TIMEOUT_SECONDS,
    MIN_TIMEOUT_SECONDS,
    ApprovedCommand,
    CommandPolicyError,
    validate_command,
)
from ..project import ProjectRoot
from .base import Tool, ToolError, ToolResult

MAX_STDOUT_CHARS = 12000
MAX_STDERR_CHARS = 12000

# How often the wait loop below wakes up to check the timeout and
# cancel_event -- small enough that Stop Task feels responsive, large
# enough not to busy-loop.
CANCEL_POLL_INTERVAL_SECONDS = 0.1
# How long to give a killed process to exit cleanly after SIGTERM before
# escalating to SIGKILL, and how long to wait for it to actually die after
# that -- generous enough for e.g. pytest's own teardown, short enough that
# Stop Task still feels responsive.
PROCESS_TERMINATE_GRACE_SECONDS = 2.0

# Allowlist, not denylist: only these survive into the subprocess's
# environment. Strips API keys, tokens, and other secrets that might
# otherwise sit in the parent process's environment and get echoed into
# captured stdout/stderr (which the model does see).
_SAFE_ENV_VARS = (
    "PATH", "HOME", "LANG", "LC_ALL", "LC_CTYPE", "TMPDIR", "SHELL",
    "VIRTUAL_ENV", "PYTHONIOENCODING",
)


class RunCommandArgs(BaseModel):
    program: str = Field(description="The executable to run, e.g. 'pytest' or 'npm'.")
    args: List[str] = Field(
        default_factory=list, description="Arguments to pass, e.g. ['-v'] or ['run', 'lint']."
    )
    timeout: int = Field(
        default=DEFAULT_TIMEOUT_SECONDS,
        ge=MIN_TIMEOUT_SECONDS,
        le=MAX_TIMEOUT_SECONDS,
        description=f"Timeout in seconds ({MIN_TIMEOUT_SECONDS}-{MAX_TIMEOUT_SECONDS}).",
    )


@dataclass(frozen=True)
class CommandExecutionResult:
    program: str
    args: List[str]
    exit_code: Optional[int]
    stdout: str
    stderr: str
    timed_out: bool
    # True if a caller-supplied cancel_event (Stop Task) interrupted this
    # command rather than it finishing or hitting its own timeout. Default
    # False keeps every existing construction of this dataclass valid.
    cancelled: bool = False


def describe_command(cmd: ApprovedCommand) -> str:
    parts = " ".join(cmd.args)
    return f"{cmd.program} {parts}".strip()


def _propose_command(project: ProjectRoot, args: RunCommandArgs) -> ToolResult:
    try:
        approved = validate_command(project, args.program, args.args, args.timeout)
    except CommandPolicyError as exc:
        raise ToolError(str(exc)) from exc

    return ToolResult(
        ok=True,
        output="(pending user approval)",
        display=f"run_command({describe_command(approved)!r})",
        pending_command=approved,
    )


def _truncate(text: str, limit: int, stream_name: str) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n[{stream_name} truncated because it exceeded the {limit}-character limit]"


def _minimal_env() -> dict:
    return {name: os.environ[name] for name in _SAFE_ENV_VARS if name in os.environ}


def _kill_process_group(proc: "subprocess.Popen") -> None:
    """Terminate the whole process group execute_command started (see
    start_new_session=True below), not just the direct child -- so e.g. a
    test runner's own worker subprocesses don't survive a timeout or Stop
    Task. SIGTERM first for a chance at clean shutdown, escalating to
    SIGKILL only if it's still alive after the grace period. Only ever
    signals the group this specific command created -- never anything else
    on the system.
    """
    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        return
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=PROCESS_TERMINATE_GRACE_SECONDS)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=PROCESS_TERMINATE_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        pass  # pragma: no cover - SIGKILL not being honored would be an OS-level issue


def execute_command(
    cmd: ApprovedCommand, cancel_event: Optional[threading.Event] = None
) -> CommandExecutionResult:
    """Actually run an approved command. Only ever called by the agent loop
    after the user has explicitly approved -- never by a tool's run().

    Runs in its own process group (start_new_session=True) and polls in
    short increments (CANCEL_POLL_INTERVAL_SECONDS) rather than blocking for
    the full timeout in one call, so that if `cancel_event` (Stop Task) is
    set while the command is running, the whole process group is actually
    terminated -- not just abandoned while the subprocess keeps running in
    the background. The existing hard per-command timeout uses the exact
    same termination path.
    """
    argv = [cmd.program, *cmd.args]
    try:
        proc = subprocess.Popen(
            argv,
            cwd=str(cmd.cwd),
            shell=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=_minimal_env(),
            start_new_session=True,
        )
    except FileNotFoundError:
        return CommandExecutionResult(
            program=cmd.program,
            args=cmd.args,
            exit_code=None,
            stdout="",
            stderr=(
                f"'{cmd.program}' is not installed or not found on PATH. Install it manually and "
                "try again -- this agent does not install dependencies automatically."
            ),
            timed_out=False,
        )
    except OSError as exc:
        return CommandExecutionResult(
            program=cmd.program,
            args=cmd.args,
            exit_code=None,
            stdout="",
            stderr=f"Failed to run '{cmd.program}': {exc}",
            timed_out=False,
        )

    start = time.monotonic()
    timed_out = False
    cancelled = False
    stdout = ""
    stderr = ""

    while True:
        remaining = cmd.timeout - (time.monotonic() - start)
        if remaining <= 0:
            timed_out = True
            _kill_process_group(proc)
            break
        if cancel_event is not None and cancel_event.is_set():
            cancelled = True
            _kill_process_group(proc)
            break
        try:
            stdout, stderr = proc.communicate(timeout=min(CANCEL_POLL_INTERVAL_SECONDS, remaining))
            break  # process finished on its own
        except subprocess.TimeoutExpired:
            continue

    if timed_out or cancelled:
        try:
            stdout, stderr = proc.communicate(timeout=PROCESS_TERMINATE_GRACE_SECONDS)
        except subprocess.TimeoutExpired:  # pragma: no cover - defense in depth
            stdout, stderr = stdout or "", stderr or ""

    return CommandExecutionResult(
        program=cmd.program,
        args=cmd.args,
        exit_code=proc.returncode,
        stdout=_truncate(stdout or "", MAX_STDOUT_CHARS, "stdout"),
        stderr=_truncate(stderr or "", MAX_STDERR_CHARS, "stderr"),
        timed_out=timed_out,
        cancelled=cancelled,
    )


def format_result_for_model(result: CommandExecutionResult) -> str:
    header = f"$ {result.program} {' '.join(result.args)}".rstrip()
    if result.cancelled:
        status = "CANCELLED BY USER (no exit code; process was terminated)"
    elif result.timed_out:
        status = "TIMED OUT (no exit code; process was terminated)"
    else:
        status = f"exit code: {result.exit_code}"
    parts = [header, status]
    if result.stdout:
        parts.append(f"--- stdout ---\n{result.stdout}")
    if result.stderr:
        parts.append(f"--- stderr ---\n{result.stderr}")
    if not result.stdout and not result.stderr:
        parts.append("(no output)")
    return "\n".join(parts)


def build_run_command_tool(project: ProjectRoot) -> Tool:
    return Tool(
        name="run_command",
        description=(
            "Propose running a local development/test command (e.g. pytest, ruff check ., "
            "mypy ., npm test, npm run <script>, python -m pytest, python manage.py test, or a "
            "project .js script via node). Only a small allowlist of programs and argument shapes "
            "is permitted; anything else is rejected before the user is ever asked. This only "
            "proposes the command: the user is shown exactly what would run and must approve it "
            "before it executes. It never installs dependencies, never touches the network, and "
            "never runs in the background."
        ),
        args_model=RunCommandArgs,
        run=lambda args: _propose_command(project, args),
    )
