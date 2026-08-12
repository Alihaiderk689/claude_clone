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
import subprocess
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


def execute_command(cmd: ApprovedCommand) -> CommandExecutionResult:
    """Actually run an approved command. Only ever called by the agent loop
    after the user has explicitly approved -- never by a tool's run().
    """
    argv = [cmd.program, *cmd.args]
    try:
        proc = subprocess.run(
            argv,
            cwd=str(cmd.cwd),
            shell=False,
            capture_output=True,
            timeout=cmd.timeout,
            encoding="utf-8",
            errors="replace",
            env=_minimal_env(),
        )
        return CommandExecutionResult(
            program=cmd.program,
            args=cmd.args,
            exit_code=proc.returncode,
            stdout=_truncate(proc.stdout or "", MAX_STDOUT_CHARS, "stdout"),
            stderr=_truncate(proc.stderr or "", MAX_STDERR_CHARS, "stderr"),
            timed_out=False,
        )
    except subprocess.TimeoutExpired as exc:
        return CommandExecutionResult(
            program=cmd.program,
            args=cmd.args,
            exit_code=None,
            stdout=_truncate(exc.stdout or "", MAX_STDOUT_CHARS, "stdout"),
            stderr=_truncate(exc.stderr or "", MAX_STDERR_CHARS, "stderr"),
            timed_out=True,
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


def format_result_for_model(result: CommandExecutionResult) -> str:
    header = f"$ {result.program} {' '.join(result.args)}".rstrip()
    if result.timed_out:
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
