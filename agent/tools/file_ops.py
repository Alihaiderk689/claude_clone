"""Approval-gated filesystem mutation tools: delete_file and rename_file.

Same propose/apply split as editing.py and tools/git.py: each tool's run()
only validates (via ../file_ops.py) and builds a ProposedFileOp describing
exactly what would happen -- neither ever touches the filesystem itself.
apply_file_op() is the only thing that ever actually calls os.remove/
os.rename, and only after the agent loop has seen the user explicitly
approve the proposal.

These exist so the model never needs run_command with rm/mv (both stay
denylisted in command_policy.py) to perform an ordinary "delete this file" /
"rename this file" request -- a structured, validated tool gives better
safety, error messages, and cache invalidation than a shell command ever
could.
"""
from __future__ import annotations

import os
from typing import Optional

from pydantic import BaseModel, Field

from ..file_ops import FileOpError, ProposedFileOp, validate_delete, validate_rename
from ..project import ProjectRoot
from .base import NotFoundError, PermissionDeniedError, Tool, ToolError, ToolResult
from .state import FileStateTracker


class DeleteFileArgs(BaseModel):
    path: str = Field(description="Existing file to delete, relative to the project root.")


class RenameFileArgs(BaseModel):
    source: str = Field(description="Existing file to rename or move, relative to the project root.")
    destination: str = Field(
        description="New path for the file, relative to the project root. Must not already exist."
    )


def _delete_file(project: ProjectRoot, args: DeleteFileArgs) -> ToolResult:
    try:
        op = validate_delete(project, args.path)
    except FileOpError as exc:
        message = str(exc)
        if "not found" in message.lower():
            raise NotFoundError(message) from exc
        if "sensitive" in message.lower():
            raise PermissionDeniedError(message) from exc
        raise ToolError(message) from exc

    return ToolResult(
        ok=True,
        output="(pending user approval)",
        display=f"delete_file({args.path!r})",
        pending_file_op=op,
    )


def _rename_file(project: ProjectRoot, args: RenameFileArgs) -> ToolResult:
    try:
        op = validate_rename(project, args.source, args.destination)
    except FileOpError as exc:
        message = str(exc)
        if "not found" in message.lower():
            raise NotFoundError(message) from exc
        if "sensitive" in message.lower():
            raise PermissionDeniedError(message) from exc
        raise ToolError(message) from exc

    return ToolResult(
        ok=True,
        output="(pending user approval)",
        display=f"rename_file({args.source!r}, {args.destination!r})",
        pending_file_op=op,
    )


def apply_file_op(op: ProposedFileOp, tracker: Optional[FileStateTracker]) -> ToolResult:
    """Actually perform an approved delete/rename. Only ever called by the
    agent loop after the user has explicitly approved -- never by a tool's
    run(). Called directly from loop.py with no Tool.execute() safety net
    (same as apply_change/apply_git_operation), so every failure mode here
    must be caught and turned into a ToolResult, never left to raise.
    """
    if op.kind == "delete":
        if not op.resolved_source.exists():
            return ToolResult(
                ok=False,
                output=f"'{op.source_path}' no longer exists -- nothing to delete.",
                error_type="StaleStateError",
                recoverable=True,
            )
        try:
            os.remove(op.resolved_source)
        except PermissionError as exc:
            return ToolResult(
                ok=False,
                output=f"Permission denied deleting '{op.source_path}': {exc}",
                error_type="PermissionError",
                recoverable=False,
            )
        except OSError as exc:
            return ToolResult(
                ok=False,
                output=f"Failed to delete '{op.source_path}': {exc}",
                error_type="ToolExecutionError",
                recoverable=True,
            )
        if tracker is not None:
            tracker.forget(op.resolved_source)
        return ToolResult(ok=True, output=f"Deleted '{op.source_path}'.")

    if op.kind == "rename":
        if not op.resolved_source.exists():
            return ToolResult(
                ok=False,
                output=f"'{op.source_path}' no longer exists -- nothing to rename.",
                error_type="StaleStateError",
                recoverable=True,
            )
        if op.resolved_destination.exists():
            return ToolResult(
                ok=False,
                output=(
                    f"'{op.destination_path}' now exists (it didn't when this rename was proposed) "
                    "-- refusing to overwrite it. Re-check and re-propose."
                ),
                error_type="StaleStateError",
                recoverable=True,
            )
        try:
            op.resolved_destination.parent.mkdir(parents=True, exist_ok=True)
            os.rename(op.resolved_source, op.resolved_destination)
        except PermissionError as exc:
            return ToolResult(
                ok=False,
                output=f"Permission denied renaming '{op.source_path}': {exc}",
                error_type="PermissionError",
                recoverable=False,
            )
        except OSError as exc:
            return ToolResult(
                ok=False,
                output=f"Failed to rename '{op.source_path}' to '{op.destination_path}': {exc}",
                error_type="ToolExecutionError",
                recoverable=True,
            )
        if tracker is not None:
            tracker.forget(op.resolved_source)
            tracker.forget(op.resolved_destination)
        return ToolResult(
            ok=True, output=f"Renamed '{op.source_path}' to '{op.destination_path}'."
        )

    return ToolResult(ok=False, output=f"Unknown file operation kind: {op.kind!r}")  # pragma: no cover


def build_delete_file_tool(project: ProjectRoot) -> Tool:
    return Tool(
        name="delete_file",
        description=(
            "Propose deleting one existing file. Only proposes it -- the user is shown the path "
            "and must approve before it's removed. Use this whenever the user asks to remove, "
            "delete, or get rid of an obsolete file. Fails if the path doesn't exist, is a "
            "directory, or matches a sensitive-file pattern (this tool never deletes directories)."
        ),
        args_model=DeleteFileArgs,
        run=lambda args: _delete_file(project, args),
    )


def build_rename_file_tool(project: ProjectRoot) -> Tool:
    return Tool(
        name="rename_file",
        description=(
            "Propose renaming or moving an existing file to a new path (moving into a different "
            "directory works the same way -- there is no separate move tool). Only proposes it -- "
            "the user is shown source -> destination and must approve before anything changes. Use "
            "this whenever the user asks to rename or move a file. Fails if the destination already "
            "exists (delete_file it first if you intend to replace it) or if either path is a "
            "directory or matches a sensitive-file pattern."
        ),
        args_model=RenameFileArgs,
        run=lambda args: _rename_file(project, args),
    )
