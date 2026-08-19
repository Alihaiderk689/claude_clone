"""Read-only filesystem tools: list_files and read_file.

Both tools resolve every path through ProjectRoot.resolve(), so neither
one can be steered outside the project root no matter what the model
sends as an argument.
"""
from __future__ import annotations

import hashlib
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
from ..task_state import TaskState
from .base import NotFoundError, PermissionDeniedError, Tool, ToolError, ToolResult
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
    force: bool = Field(
        default=False,
        description=(
            "Set true to force a fresh full read even if this file was already read earlier in "
            "this conversation and hasn't changed since. Not needed normally -- an unchanged "
            "file's content is already available from the earlier read."
        ),
    )


def _safe_resolve(project: ProjectRoot, path: str) -> Path:
    try:
        return project.resolve(path)
    except PathSecurityError as exc:
        raise ToolError(str(exc)) from exc


def _list_files(project: ProjectRoot, args: ListFilesArgs) -> ToolResult:
    target = _safe_resolve(project, args.path)

    try:
        exists = target.exists()
    except PermissionError as exc:
        raise PermissionDeniedError(f"Permission denied accessing '{args.path}': {exc}") from exc
    if not exists:
        raise NotFoundError(f"Path not found: '{args.path}'")
    if not target.is_dir():
        raise ToolError(f"'{args.path}' is a file, not a directory. Use read_file instead.")

    entries: List[str] = []
    truncated = False

    def _reraise(err: OSError) -> None:
        # os.walk's default onerror=None silently skips a directory it
        # can't scan (e.g. permission denied), which would otherwise make
        # list_files lie and report a locked directory as merely empty --
        # actively hiding the real problem from the model instead of
        # reporting it. Re-raise so the except clause below handles it.
        raise err

    try:
        for current_root, dirnames, filenames in os.walk(target, onerror=_reraise):
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
    except PermissionError as exc:
        raise PermissionDeniedError(f"Permission denied listing '{args.path}': {exc}") from exc

    output = "\n".join(entries) if entries else "(empty directory)"
    if truncated:
        output += (
            f"\n... truncated at {args.max_entries} entries. "
            "Narrow `path` to a subdirectory to see more."
        )

    return ToolResult(ok=True, output=output, display=f"list_files({args.path!r})")


def _read_file(
    project: ProjectRoot,
    tracker: Optional[FileStateTracker],
    args: ReadFileArgs,
    task_state: Optional[TaskState] = None,
) -> ToolResult:
    target = _safe_resolve(project, args.path)

    try:
        exists = target.exists()
    except PermissionError as exc:
        raise PermissionDeniedError(f"Permission denied accessing '{args.path}': {exc}") from exc
    if not exists:
        raise NotFoundError(f"File not found: '{args.path}'")
    if target.is_dir():
        raise ToolError(f"'{args.path}' is a directory, not a file. Use list_files instead.")
    if project.is_sensitive(target):
        raise PermissionDeniedError(f"Refusing to read '{args.path}': it matches a sensitive-file pattern.")

    if args.start_line is not None and args.end_line is not None and args.end_line < args.start_line:
        raise ToolError("end_line must be greater than or equal to start_line.")

    try:
        size = target.stat().st_size
    except PermissionError as exc:
        raise PermissionDeniedError(f"Permission denied accessing '{args.path}': {exc}") from exc
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
    except PermissionError as exc:
        raise PermissionDeniedError(f"Permission denied reading '{args.path}': {exc}") from exc

    # Phase 9 unchanged-file short-circuit: check freshness against the
    # PREVIOUSLY recorded hash before it gets overwritten below -- this is
    # what lets it also catch an external modification (disk content
    # changed without going through edit_file/write_file, so files_inspected
    # alone wouldn't know), not just "hasn't been edited by this agent."
    # Full reads only: a ranged read's cache semantics would be ambiguous
    # (was this exact range shown before, or a different one?).
    is_cache_hit = (
        not wants_range
        and not args.force
        and task_state is not None
        and tracker is not None
        and args.path in task_state.files_inspected
        and tracker.is_fresh(target, text)
    )

    # Record against the full content regardless of range, so a later
    # edit_file call can detect drift anywhere in the file, not just in
    # whatever slice was shown here.
    if tracker is not None:
        tracker.record(target, text)

    if is_cache_hit:
        # Structured, not conversational: a short "reuse it" sentence read
        # badly to a small model mid-task (observed live -- it interpreted
        # the notice as the task being over and stopped to ask what to do
        # next instead of continuing). Framing this as internal state (path/
        # status/line count) with an explicit continuation instruction, same
        # shape as a tool status rather than a reply to a question, measurably
        # reduces that misreading. The full content deliberately isn't
        # resent here -- it's still intact earlier in this conversation
        # (context_manager.compact_messages never supersedes it with this
        # stub, see its cache_hit-aware pass) -- so this stays cheap on
        # every repeat read instead of paying the full content's context
        # cost again; _looks_like_unactioned_narration in loop.py is the
        # real backstop if a small model still stalls anyway.
        short_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
        line_count = len(text.splitlines())
        return ToolResult(
            ok=True,
            output=(
                f"path={args.path!r} status=unchanged lines={line_count} hash={short_hash}\n"
                "Content is identical to your earlier read_file result for this path above in this "
                "conversation -- use that content, nothing new to see here. The task is NOT done "
                "yet: proceed immediately with the user's request (e.g. call edit_file/write_file/"
                "delete_file/rename_file) rather than asking what to do next. Pass force=true only "
                "if you specifically suspect it's stale."
            ),
            display=f"read_file({args.path!r}) [cached]",
            cache_hit=True,
        )

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


def build_read_file_tool(
    project: ProjectRoot,
    tracker: Optional[FileStateTracker] = None,
    task_state: Optional[TaskState] = None,
) -> Tool:
    return Tool(
        name="read_file",
        description=(
            "Read the contents of a single file inside the project, relative to the project root. "
            "Returns line-numbered text. For large files, pass start_line/end_line to read a "
            "portion instead of guessing; the tool will tell you if a full read is too large. If "
            "you already read this exact file earlier and it hasn't changed, this returns a short "
            "'unchanged' notice instead of the full content again -- reuse what you already have."
        ),
        args_model=ReadFileArgs,
        run=lambda args: _read_file(project, tracker, args, task_state),
    )
