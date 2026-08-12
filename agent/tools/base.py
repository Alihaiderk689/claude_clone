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


class ToolError(Exception):
    """Raised by tool implementations for expected, user-facing failures
    (bad path, file too large, no matches, etc). The message is safe to
    send back to the model as-is.
    """


@dataclass(frozen=True)
class ToolResult:
    ok: bool
    output: str  # text sent back to the model as the tool's response
    display: Optional[str] = None  # short human-readable summary for the CLI
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
            return ToolResult(ok=False, output=f"Invalid arguments for {self.name}: {exc}")

        try:
            return self.run(args)
        except ToolError as exc:
            return ToolResult(ok=False, output=str(exc))
        except Exception as exc:  # pragma: no cover - safety net; tools should raise ToolError
            return ToolResult(ok=False, output=f"Unexpected error running {self.name}: {exc}")
