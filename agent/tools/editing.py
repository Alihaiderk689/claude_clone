"""Approval-gated file modification tools: edit_file and write_file.

Neither tool ever writes to disk on its own. Their run() functions only
validate the request and build a ProposedChange (including the exact diff
the user will see) entirely in memory. apply_change() performs the actual
write, and the agent loop (agent/loop.py) only ever calls it after the user
has explicitly approved the proposed change — never on the model's say-so
alone. This is the whole point of Phase 3: the model proposes, Python
validates and writes, the user approves in between.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Optional, Tuple

from pydantic import BaseModel, Field

from ..diff import ProposedChange
from ..project import HARD_MAX_FILE_SIZE_BYTES, PathSecurityError, ProjectRoot
from .base import (
    NotFoundError,
    PermissionDeniedError,
    StaleStateError,
    Tool,
    ToolError,
    ToolResult,
)
from .state import FileStateTracker

DEFAULT_NEW_FILE_MODE = 0o644


class EditFileArgs(BaseModel):
    path: str = Field(description="Existing file to edit, relative to the project root.")
    old_text: str = Field(
        description="Exact existing text to replace. Must appear exactly once in the file."
    )
    new_text: str = Field(description="Text to replace old_text with.")


class WriteFileArgs(BaseModel):
    path: str = Field(description="New file to create, relative to the project root.")
    content: str = Field(description="Full content of the new file.")


def _safe_resolve(project: ProjectRoot, path: str) -> Path:
    try:
        return project.resolve(path)
    except PathSecurityError as exc:
        raise ToolError(str(exc)) from exc


def _decode_and_detect_newline(raw: bytes) -> Tuple[str, str]:
    """Decode file bytes to \\n-normalized text (matching what read_file shows
    the model) and detect the file's dominant newline style, so it can be
    restored on write instead of silently converting the whole file.
    """
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise ToolError("This looks like a binary file and can't be edited as text.")
    crlf_count = text.count("\r\n")
    lf_only_count = text.count("\n") - crlf_count
    newline = "\r\n" if crlf_count > lf_only_count else "\n"
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    return normalized, newline


def _edit_file(
    project: ProjectRoot, tracker: Optional[FileStateTracker], args: EditFileArgs
) -> ToolResult:
    target = _safe_resolve(project, args.path)

    try:
        exists = target.exists()
    except PermissionError as exc:
        raise PermissionDeniedError(f"Permission denied accessing '{args.path}': {exc}") from exc
    if not exists:
        raise NotFoundError(f"File not found: '{args.path}'. Use write_file to create a new file.")
    if target.is_dir():
        raise ToolError(f"'{args.path}' is a directory, not a file.")
    if project.is_sensitive(target):
        raise PermissionDeniedError(f"Refusing to edit '{args.path}': it matches a sensitive-file pattern.")
    if not args.old_text:
        raise ToolError("old_text must not be empty.")
    if args.old_text == args.new_text:
        raise ToolError("old_text and new_text are identical; there's nothing to change.")

    try:
        size = target.stat().st_size
    except PermissionError as exc:
        raise PermissionDeniedError(f"Permission denied accessing '{args.path}': {exc}") from exc
    if size > HARD_MAX_FILE_SIZE_BYTES:
        raise ToolError(f"'{args.path}' is {size:,} bytes, far too large to edit.")

    try:
        raw = target.read_bytes()
    except PermissionError as exc:
        raise PermissionDeniedError(f"Permission denied reading '{args.path}': {exc}") from exc

    normalized, newline = _decode_and_detect_newline(raw)

    if tracker is not None and not tracker.is_fresh(target, normalized):
        raise StaleStateError(
            f"'{args.path}' changed on disk since it was last read. Refusing to apply a possibly "
            "stale edit — read_file it again and re-propose the edit against its current content."
        )

    occurrences = normalized.count(args.old_text)
    if occurrences == 0:
        raise ToolError(
            f"Target text not found in '{args.path}'. Re-read the file and provide an exact match "
            "for old_text (whitespace and indentation must match exactly)."
        )
    if occurrences > 1:
        raise ToolError(
            f"old_text appears {occurrences} times in '{args.path}', so the edit is ambiguous. "
            "Include more surrounding context in old_text so it matches exactly once."
        )

    new_normalized = normalized.replace(args.old_text, args.new_text, 1)
    new_size = len(new_normalized.encode("utf-8"))
    if new_size > HARD_MAX_FILE_SIZE_BYTES:
        raise ToolError(f"The edited file would be {new_size:,} bytes, far too large.")

    change = ProposedChange(
        path=args.path,
        resolved_path=target,
        kind="edit",
        old_content=normalized,
        new_content=new_normalized,
        newline=newline,
        original_mode=target.stat().st_mode,
    )
    return ToolResult(
        ok=True,
        output="(pending user approval)",
        display=f"edit_file({args.path!r})",
        pending_change=change,
    )


def _write_file(
    project: ProjectRoot, tracker: Optional[FileStateTracker], args: WriteFileArgs
) -> ToolResult:
    target = _safe_resolve(project, args.path)

    if project.is_sensitive(target):
        raise PermissionDeniedError(f"Refusing to create '{args.path}': it matches a sensitive-file pattern.")
    if target.is_dir():
        raise ToolError(f"'{args.path}' is a directory.")
    try:
        already_exists = target.exists()
    except PermissionError as exc:
        raise PermissionDeniedError(f"Permission denied accessing '{args.path}': {exc}") from exc
    if already_exists:
        raise ToolError(
            f"'{args.path}' already exists. Use edit_file to modify an existing file instead of "
            "write_file, which is only for creating new files."
        )

    content_size = len(args.content.encode("utf-8"))
    if content_size > HARD_MAX_FILE_SIZE_BYTES:
        raise ToolError(f"The new file would be {content_size:,} bytes, far too large.")

    change = ProposedChange(
        path=args.path,
        resolved_path=target,
        kind="create",
        old_content=None,
        new_content=args.content,
        newline="\n",
        original_mode=None,
    )
    return ToolResult(
        ok=True,
        output="(pending user approval)",
        display=f"write_file({args.path!r})",
        pending_change=change,
    )


def _atomic_write(path: Path, content: str, newline: str, mode: Optional[int]) -> None:
    """Write `content` to `path` atomically: build the full new content in a
    temp file in the same directory (so the final os.replace is an atomic
    rename on the same filesystem), fsync it, restore the original newline
    style and permissions, then swap it into place. If anything goes wrong
    before the final replace, the original file is untouched and the temp
    file is cleaned up.
    """
    raw_content = content if newline == "\n" else content.replace("\n", newline)
    directory = path.parent
    directory.mkdir(parents=True, exist_ok=True)

    fd, tmp_name = tempfile.mkstemp(dir=str(directory), prefix=f".{path.name}.", suffix=".tmp")
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(raw_content.encode("utf-8"))
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp_path, mode if mode is not None else DEFAULT_NEW_FILE_MODE)
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def apply_change(change: ProposedChange, tracker: Optional[FileStateTracker]) -> ToolResult:
    """Actually write an approved ProposedChange to disk, then verify the
    result before reporting success. Only ever called by the agent loop
    after the user has explicitly approved — never by a tool's run().
    """
    try:
        _atomic_write(change.resolved_path, change.new_content, change.newline, change.original_mode)
    except PermissionError as exc:
        return ToolResult(
            ok=False,
            output=f"Permission denied writing '{change.path}': {exc}. The original file was left unchanged.",
            error_type="PermissionError",
            recoverable=False,
        )
    except OSError as exc:
        return ToolResult(
            ok=False,
            output=f"Failed to write '{change.path}': {exc}. The original file was left unchanged.",
            error_type="ToolExecutionError",
            recoverable=True,
        )

    try:
        written_text, _ = _decode_and_detect_newline(change.resolved_path.read_bytes())
    except (OSError, ToolError) as exc:
        return ToolResult(
            ok=False,
            output=f"Wrote '{change.path}' but could not verify it: {exc}",
            error_type="ToolExecutionError",
            recoverable=True,
        )

    if written_text != change.new_content:
        return ToolResult(
            ok=False,
            output=(
                f"Wrote '{change.path}' but the content on disk doesn't match what was approved. "
                "Read the file again before proposing further edits."
            ),
        )

    if tracker is not None:
        tracker.record(change.resolved_path, written_text)

    verb = "Created" if change.kind == "create" else "Updated"
    return ToolResult(
        ok=True,
        output=(
            f"{verb} '{change.path}' successfully. The user approved this change and it has been "
            "written to disk and verified."
        ),
    )


def build_edit_file_tool(project: ProjectRoot, tracker: Optional[FileStateTracker] = None) -> Tool:
    return Tool(
        name="edit_file",
        description=(
            "Propose replacing an exact block of text in an existing file with new text. "
            "old_text must match the file's current content exactly once — include enough "
            "surrounding context to make the match unambiguous. This only proposes the change: "
            "the user is shown a diff and must approve it before anything is written. Read the "
            "file first with read_file so old_text matches its real, current content."
        ),
        args_model=EditFileArgs,
        run=lambda args: _edit_file(project, tracker, args),
    )


def build_write_file_tool(project: ProjectRoot, tracker: Optional[FileStateTracker] = None) -> Tool:
    return Tool(
        name="write_file",
        description=(
            "Propose creating a brand-new file with the given content. Fails if the file already "
            "exists — use edit_file to modify existing files. This only proposes the change: the "
            "user is shown the new file's content as a diff and must approve it before it's created."
        ),
        args_model=WriteFileArgs,
        run=lambda args: _write_file(project, tracker, args),
    )
