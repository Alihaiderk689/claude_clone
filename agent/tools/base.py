"""Base building blocks for the tool system.

A Tool pairs a Pydantic argument schema with a plain Python function.
The schema gives us both request-time validation and, for free, the JSON
schema Ollama needs to advertise the tool to the model.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Type

from pydantic import BaseModel, ValidationError

from ..command_policy import ApprovedCommand
from ..diff import ProposedChange
from ..git_policy import ProposedGitOperation
from ..planner import Plan


class ToolError(Exception):
    """Raised by tool implementations for expected, user-facing failures
    (bad path, file too large, no matches, etc). The message is safe to
    send back to the model as-is.

    Subclass this for failures worth distinguishing (see the taxonomy below)
    -- `error_type`/`recoverable` class attributes are read by
    `Tool.execute()` and copied onto the resulting `ToolResult` so callers
    (loop.py's repetition/failed-approach tracking, logging, the VS Code UI)
    can react to *what kind* of failure this was, not just that it failed.
    Anything raised as a bare `ToolError` (most existing call sites) is
    still handled safely -- it just falls back to the generic
    "ToolExecutionError"/recoverable=True classification below.
    """

    error_type: str = "ToolExecutionError"
    recoverable: bool = True


class ValidationFailedError(ToolError):
    """The request was structurally fine but rejected on its merits (bad
    argument combination, empty required text, ambiguous match, etc) --
    the model should adjust its arguments and retry, not give up."""

    error_type = "ValidationError"
    recoverable = True


class NotFoundError(ToolError):
    """A referenced file/directory/path does not exist. Recoverable: the
    model should re-inspect the project (list_files/search_files) rather
    than guessing the same path again."""

    error_type = "NotFoundError"
    recoverable = True


class PermissionDeniedError(ToolError):
    """The OS denied access (sensitive-file protection, filesystem
    permissions). Not meaningfully recoverable by retrying -- there is no
    tool-level workaround, only a different request from the user."""

    error_type = "PermissionError"
    recoverable = False


class StaleStateError(ToolError):
    """The target changed since the model last inspected it (stale
    read-before-edit, staged files changed since a commit was proposed).
    Recoverable: re-inspect, then retry."""

    error_type = "StaleStateError"
    recoverable = True


class ToolTimeoutError(ToolError):
    """A tool's own subprocess (Git, etc) exceeded its timeout."""

    error_type = "TimeoutError"
    recoverable = True


class ExternalToolUnavailableError(ToolError):
    """A required external program (git, ...) is not installed or not on
    PATH. Not recoverable by retrying -- the user has to fix their
    environment."""

    error_type = "ExternalToolUnavailableError"
    recoverable = False


@dataclass(frozen=True)
class ToolResult:
    ok: bool
    output: str  # text sent back to the model as the tool's response
    display: Optional[str] = None  # short human-readable summary for the CLI
    # Populated on failure (ok=False) from the raised exception's
    # error_type/recoverable -- see the ToolError taxonomy above. None for
    # a successful result, or for a ToolResult built by older/hand-written
    # code that predates this field; treat None as "unclassified", not as
    # a signal of anything in particular.
    error_type: Optional[str] = None
    recoverable: Optional[bool] = None
    # Set only by edit_file/write_file on success: a validated, not-yet-applied
    # change. The agent loop must pause for user approval before anything is
    # written — see diff.apply_change() and loop.py's "confirm" event.
    pending_change: Optional[ProposedChange] = None
    # Set only by run_command on success: a validated, not-yet-executed command.
    # The agent loop must pause for user approval before it runs — see
    # tools/terminal.py's execute_command() and loop.py's "confirm_command" event.
    pending_command: Optional[ApprovedCommand] = None
    # Set only by git_create_branch/git_stage/git_commit on success: a validated,
    # not-yet-executed Git operation. The agent loop must pause for user approval
    # first — see tools/git.py's apply_git_operation() and loop.py's
    # "confirm_git_operation" event.
    pending_git_operation: Optional[ProposedGitOperation] = None
    # Set only by create_plan on success: a proposed plan not yet adopted into
    # the task state. The agent loop pauses for user approval — see loop.py's
    # "confirm_plan" event. Unlike the other pending_* kinds this defaults to
    # approve on a bare Enter (a plan has no side effects of its own; approving
    # it is not permission to skip file/command/Git approvals).
    pending_plan: Optional[Plan] = None


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    args_model: Type[BaseModel]
    run: Callable[[BaseModel], ToolResult]

    def to_ollama_schema(self) -> Dict[str, Any]:
        parameters = self.args_model.model_json_schema()
        parameters.pop("title", None)
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": parameters,
            },
        }

    def execute(self, raw_arguments: Optional[Dict[str, Any]]) -> ToolResult:
        try:
            args = self.args_model.model_validate(raw_arguments or {})
        except ValidationError as exc:
            return ToolResult(
                ok=False,
                output=f"Invalid arguments for {self.name}: {exc}",
                error_type="ValidationError",
                recoverable=True,
            )

        try:
            return self.run(args)
        except ToolError as exc:
            return ToolResult(
                ok=False, output=str(exc), error_type=exc.error_type, recoverable=exc.recoverable
            )
        except Exception as exc:  # pragma: no cover - safety net; tools should raise ToolError
            return ToolResult(
                ok=False,
                output=f"Unexpected error running {self.name}: {exc}",
                error_type="InternalError",
                recoverable=False,
            )
