"""Read-only filesystem tools: list_files and read_file.

Both tools resolve every path through ProjectRoot.resolve(), so neither
one can be steered outside the project root no matter what the model
sends as an argument.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field

from ..project import (
    HARD_MAX_FILE_SIZE_BYTES,
    MAX_FILE_LINES,
    MAX_FILE_SIZE_BYTES,
    MAX_LIST_ENTRIES,
    PathSecurityError,
    ProjectRoot,
)
from .base import Tool, ToolError, ToolResult
from .state import FileStateTracker


class ListFilesArgs(BaseModel):
    path: str = Field(
        default=".",
        description="Directory to list, relative to the project root. Defaults to the root itself.",
    )
    max_entries: int = Field(
        default=200,
        ge=1,
        le=MAX_LIST_ENTRIES,
        description="Maximum number of files/directories to return.",
    )


class ReadFileArgs(BaseModel):
    path: str = Field(description="File path to read, relative to the project root.")
    start_line: Optional[int] = Field(
        default=None, ge=1, description="First line to read, 1-indexed and inclusive."
    )
    end_line: Optional[int] = Field(
        default=None, ge=1, description="Last line to read, 1-indexed and inclusive."
    )


def _safe_resolve(project: ProjectRoot, path: str) -> Path:
    try:
        return project.resolve(path)
    except PathSecurityError as exc:
        raise ToolError(str(exc)) from exc


def _list_files(project: ProjectRoot, args: ListFilesArgs) -> ToolResult:
    target = _safe_resolve(project, args.path)

    if not target.exists():
        raise ToolError(f"Path not found: '{args.path}'")
    if not target.is_dir():
        raise ToolError(f"'{args.path}' is a file, not a directory. Use read_file instead.")

    entries: List[str] = []
    truncated = False

    for current_root, dirnames, filenames in os.walk(target):
        dirnames[:] = sorted(d for d in dirnames if not project.is_ignored_dir(d))
        filenames = sorted(filenames)

        rel_dir = Path(current_root).relative_to(project.root)

        for d in dirnames:
            if len(entries) >= args.max_entries:
                truncated = True
                break
            rel = d if rel_dir == Path(".") else str(rel_dir / d)
            entries.append(rel + "/")
        if truncated:
            break

        for f in filenames:
            if project.is_sensitive(Path(f)):
                continue
            if len(entries) >= args.max_entries:
                truncated = True
                break
            rel = f if rel_dir == Path(".") else str(rel_dir / f)
            entries.append(rel)
        if truncated:
            break

    output = "\n".join(entries) if entries else "(empty directory)"
    if truncated:
        output += (
            f"\n... truncated at {args.max_entries} entries. "
            "Narrow `path` to a subdirectory to see more."
        )

    return ToolResult(ok=True, output=output, display=f"list_files({args.path!r})")


def _read_file(
    project: ProjectRoot, tracker: Optional[FileStateTracker], args: ReadFileArgs
) -> ToolResult:
    target = _safe_resolve(project, args.path)

    if not target.exists():
        raise ToolError(f"File not found: '{args.path}'")
    if target.is_dir():
        raise ToolError(f"'{args.path}' is a directory, not a file. Use list_files instead.")
    if project.is_sensitive(target):
        raise ToolError(f"Refusing to read '{args.path}': it matches a sensitive-file pattern.")

    if args.start_line is not None and args.end_line is not None and args.end_line < args.start_line:
        raise ToolError("end_line must be greater than or equal to start_line.")

    size = target.stat().st_size
    if size > HARD_MAX_FILE_SIZE_BYTES:
        raise ToolError(
            f"'{args.path}' is {size:,} bytes, far too large to read even with a line range."
        )

    wants_range = args.start_line is not None or args.end_line is not None
    if not wants_range and size > MAX_FILE_SIZE_BYTES:
        raise ToolError(
            f"'{args.path}' is {size:,} bytes, over the {MAX_FILE_SIZE_BYTES:,}-byte limit for a full "
            "read. Retry with start_line/end_line to read part of the file."
        )

    try:
        text = target.read_text(encoding="utf-8")
    except (UnicodeDecodeError, ValueError):
        raise ToolError(f"'{args.path}' looks like a binary file and can't be displayed as text.")

    # Record against the full content regardless of range, so a later
    # edit_file call can detect drift anywhere in the file, not just in
    # whatever slice was shown here.
    if tracker is not None:
        tracker.record(target, text)

    lines = text.splitlines()
    total_lines = len(lines)

    if wants_range:
        start = max(1, args.start_line or 1)
        end = min(total_lines, args.end_line or total_lines)
        if total_lines == 0:
            raise ToolError(f"'{args.path}' is empty.")
        if start > total_lines:
            raise ToolError(f"start_line {start} is beyond the end of the file ({total_lines} lines).")
        selected = lines[start - 1 : end]
        header = f"{args.path} (lines {start}-{end} of {total_lines})"
        body = "\n".join(f"{i:>5} | {line}" for i, line in enumerate(selected, start=start))
        output = f"{header}\n{body}"
        return ToolResult(ok=True, output=output, display=f"read_file({args.path!r})")

    if total_lines > MAX_FILE_LINES:
        selected = lines[:MAX_FILE_LINES]
        header = f"{args.path} ({total_lines} lines, showing 1-{MAX_FILE_LINES})"
        body = "\n".join(f"{i:>5} | {line}" for i, line in enumerate(selected, start=1))
        output = f"{header}\n{body}\n... truncated. Use start_line/end_line to read the rest."
        return ToolResult(ok=True, output=output, display=f"read_file({args.path!r})")

    header = f"{args.path} ({total_lines} lines)"
    body = "\n".join(f"{i:>5} | {line}" for i, line in enumerate(lines, start=1))
    output = f"{header}\n{body}"
    return ToolResult(ok=True, output=output, display=f"read_file({args.path!r})")


def build_list_files_tool(project: ProjectRoot) -> Tool:
    return Tool(
        name="list_files",
        description=(
            "List files and directories inside the project, relative to the project root. "
            "Irrelevant directories (.git, node_modules, __pycache__, .venv, dist, build, etc.) "
            "are skipped automatically. Use this first to understand the project's structure."
        ),
        args_model=ListFilesArgs,
        run=lambda args: _list_files(project, args),
    )


def build_read_file_tool(project: ProjectRoot, tracker: Optional[FileStateTracker] = None) -> Tool:
    return Tool(
        name="read_file",
        description=(
            "Read the contents of a single file inside the project, relative to the project root. "
            "Returns line-numbered text. For large files, pass start_line/end_line to read a "
            "portion instead of guessing; the tool will tell you if a full read is too large."
        ),
        args_model=ReadFileArgs,
        run=lambda args: _read_file(project, tracker, args),
    )
