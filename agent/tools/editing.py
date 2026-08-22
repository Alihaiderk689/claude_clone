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

import ast
import os
import re
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

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
    ValidationFailedError,
)
from .state import FileStateTracker

DEFAULT_NEW_FILE_MODE = 0o644

# Same live-observed failure as the path-placeholder guard in project.py,
# one level up the call: instead of a placeholder *path*, the model sends a
# placeholder *argument value* -- e.g. old_text='<existing text to replace>'
# or new_text='<new content here>' -- after already having the file's real
# content in front of it from an earlier read_file call. Unlike a path,
# arbitrary code legitimately contains '<'/'>' (comparisons, generics,
# HTML), so the check here is deliberately narrower than project.py's: it
# only matches when the *entire* argument, trimmed, is nothing but one
# bracketed phrase -- real old_text/new_text (even a single short line) is
# never shaped like that.
_PLACEHOLDER_TOKEN_RE = re.compile(r"^<[^<>\n]{1,200}>$")


def _looks_like_placeholder_token(text: str) -> bool:
    return bool(_PLACEHOLDER_TOKEN_RE.match(text.strip()))


# read_file renders a file as "  12 | def foo():" -- the line number and pipe
# are display furniture, not part of the file. A model told to copy old_text
# "verbatim from the read_file output" will often copy that prefix too, and
# then nothing matches, because the prefix is not in the file on disk. This
# was the single largest source of "Target text not found" failures with a
# small local model. Rather than only documenting the rule (see the tool
# description and the system prompt, both updated), strip the prefix
# defensively -- but only as a *fallback* after a literal match has already
# failed, and only when EVERY non-blank line carries it, which is the
# signature of a copy-paste from read_file and not of real source code.
_LINE_NUMBER_PREFIX_RE = re.compile(r"^\s*\d+\s*\|\s?")


def _strip_line_number_prefixes(text: str) -> str:
    """Remove read_file's 'NNN | ' prefix from every line, or return `text`
    unchanged if it doesn't uniformly look like line-numbered read_file
    output. Never partially strips: it's all lines or none."""
    lines = text.split("\n")
    meaningful = [line for line in lines if line.strip()]
    if not meaningful:
        return text
    if not all(_LINE_NUMBER_PREFIX_RE.match(line) for line in meaningful):
        return text
    return "\n".join(
        _LINE_NUMBER_PREFIX_RE.sub("", line) if line.strip() else line for line in lines
    )


def _occurrence_lines(haystack: str, needle: str, limit: int = 10) -> List[int]:
    """1-indexed line numbers where `needle` starts in `haystack`. Reported
    back to the model on an ambiguous match so it can add context around a
    specific occurrence instead of guessing which one is which."""
    found: List[int] = []
    idx = haystack.find(needle)
    while idx != -1 and len(found) < limit:
        found.append(haystack.count("\n", 0, idx) + 1)
        idx = haystack.find(needle, idx + 1)
    return found


def _as_block(text: str) -> List[str]:
    """Split into lines, dropping the empty trailing element a trailing
    newline produces -- whitespace-insensitive matching works on whole
    lines, so a trailing "\n" must not demand an extra blank line."""
    lines = text.split("\n")
    if len(lines) > 1 and lines[-1] == "":
        lines = lines[:-1]
    return lines


def _dedent_block(lines: List[str]) -> List[str]:
    """Strip the common leading indentation off a block, preserving the
    relative nesting inside it. Blank lines become truly empty."""
    indents = [len(line) - len(line.lstrip()) for line in lines if line.strip()]
    base = min(indents) if indents else 0
    return [line[base:] if line.strip() else "" for line in lines]


def _apply_literal(
    path: str, normalized: str, old_text: str, new_text: str, count: int, replace_all: bool
) -> str:
    if count > 1 and not replace_all:
        where = ", ".join(str(n) for n in _occurrence_lines(normalized, old_text))
        raise ToolError(
            f"old_text appears {count} times in '{path}' (starting at lines {where}), so the edit "
            "is ambiguous. Either include more surrounding context in old_text so it matches "
            "exactly once, or pass replace_all=true to change every occurrence."
        )
    return normalized.replace(old_text, new_text, -1 if replace_all else 1)


def _apply_whitespace_insensitive(
    path: str, normalized: str, old_text: str, new_text: str, replace_all: bool
) -> str:
    """Last-resort match that compares lines by their stripped content, then
    re-applies the file's own indentation to the replacement.

    This is what rescues an edit whose only defect is indentation drift --
    a small model reproducing a block it read but flattening or shifting the
    leading whitespace. Only reached after both literal strategies fail, and
    still refuses an ambiguous match, so it loosens *whitespace* matching
    without loosening the "exactly one target" guarantee.
    """
    file_lines = normalized.split("\n")
    old_lines = _as_block(old_text)
    new_lines = _as_block(new_text)

    span = len(old_lines)
    target = [line.strip() for line in old_lines]
    starts = [
        i
        for i in range(len(file_lines) - span + 1)
        if [line.strip() for line in file_lines[i : i + span]] == target
    ]

    if not starts:
        raise ToolError(
            f"Target text not found in '{path}'. Three things to check, in order: (1) old_text "
            "must be the file's real content -- do NOT include read_file's 'NNN | ' line-number "
            "prefixes, those are display only and are not in the file; (2) copy a real, "
            "contiguous block from your most recent read_file result for this path, not a "
            "paraphrase; (3) if you want to replace the file's entire content rather than one "
            "block, call write_file with overwrite=true and the complete new content instead."
        )
    if len(starts) > 1 and not replace_all:
        where = ", ".join(str(i + 1) for i in starts)
        raise ToolError(
            f"old_text matches {len(starts)} places in '{path}' (ignoring indentation; at lines "
            f"{where}), so the edit is ambiguous. Either include more surrounding context in "
            "old_text so it matches exactly once, or pass replace_all=true to change every "
            "occurrence."
        )

    dedented_new = _dedent_block(new_lines)
    for start in reversed(starts if replace_all else starts[:1]):
        matched = file_lines[start : start + span]
        # Anchor on the shallowest matched line rather than the first one:
        # identical for a block that starts at its own outermost level (the
        # common case), but correct for one that starts deeper than it ends.
        widths = [len(line) - len(line.lstrip()) for line in matched if line.strip()]
        base_indent = " " * (min(widths) if widths else 0)
        file_lines[start : start + span] = [
            base_indent + line if line else "" for line in dedented_new
        ]
    return "\n".join(file_lines)


def _resolve_replacement(
    path: str, normalized: str, old_text: str, new_text: str, replace_all: bool
) -> str:
    """Produce the file's new content, trying progressively more forgiving
    match strategies. Order matters: an exact literal match always wins, so
    neither fallback can change the result of an edit that was already
    correct -- they only turn a hard failure into a success.
    """
    # 1. Exact literal match, exactly as before.
    count = normalized.count(old_text)
    if count:
        return _apply_literal(path, normalized, old_text, new_text, count, replace_all)

    # 2. Same, after stripping read_file's line-number prefixes.
    stripped_old = _strip_line_number_prefixes(old_text)
    stripped_new = _strip_line_number_prefixes(new_text)
    if stripped_old != old_text:
        count = normalized.count(stripped_old)
        if count:
            if stripped_old == stripped_new:
                raise ValidationFailedError(
                    "old_text and new_text are identical once read_file's 'NNN | ' line-number "
                    "prefixes are removed, so there's nothing to change. Those prefixes are "
                    "display only -- write new_text as the real replacement code."
                )
            return _apply_literal(path, normalized, stripped_old, stripped_new, count, replace_all)

    # 3. Whitespace-insensitive line matching, on the best variant available.
    return _apply_whitespace_insensitive(
        path, normalized, stripped_old, stripped_new, replace_all
    )


# A model can pass every content-shape check above (real text, not a
# placeholder) and still produce Python that doesn't parse -- observed live:
# qwen2.5-coder:3b generated a bubble_sort() body where every line, at every
# nesting depth (function body, for loop, nested for loop, if statement),
# used the exact same single-space indent, which is a genuine IndentationError,
# not a style nitpick. Nothing before this point ever ran the proposed content
# through a real Python parser, so that would have been shown to the user as
# an approvable diff and wrote a syntax-broken file on approval -- "the tool
# succeeded" and "the file is valid Python" are not the same claim, and only
# the second one actually matters. ast.parse is stdlib (no new dependency,
# consistent with this project's "don't add a dependency without reason"
# rule) and catches IndentationError/TabError too, since both subclass
# SyntaxError. Scoped to .py files only -- this agent's tools are otherwise
# language-agnostic and have no general syntax checker for anything else.
def _validate_python_syntax(path: str, content: str) -> None:
    if not path.endswith(".py"):
        return
    try:
        ast.parse(content, filename=path)
    except SyntaxError as exc:
        location = f"line {exc.lineno}" + (f", column {exc.offset}" if exc.offset else "")
        offending_line = (exc.text or "").rstrip("\n")
        detail = f"\n    {offending_line}" if offending_line.strip() else ""
        raise ValidationFailedError(
            f"The resulting content for '{path}' is not valid Python: {exc.msg} ({location}).{detail}\n"
            "This is usually an indentation mistake (e.g. every nested line using the same "
            "indent instead of increasing it for each nested block). Fix the syntax and propose "
            "the edit again -- nothing was written."
        ) from exc


class EditFileArgs(BaseModel):
    path: str = Field(description="Existing file to edit, relative to the project root.")
    old_text: str = Field(
        description=(
            "Existing text to replace, copied from the file's real content. Do NOT include the "
            "'NNN | ' line-number prefixes read_file adds for display -- they are not part of the "
            "file. Must identify exactly one place unless replace_all is set."
        )
    )
    new_text: str = Field(description="Text to replace old_text with.")
    replace_all: bool = Field(
        default=False,
        description=(
            "Set true to replace every occurrence of old_text instead of failing when it appears "
            "more than once. Leave false (the default) when you mean one specific place."
        ),
    )


class WriteFileArgs(BaseModel):
    path: str = Field(description="File to create, relative to the project root.")
    content: str = Field(description="Full content of the file.")
    overwrite: bool = Field(
        default=False,
        description=(
            "Set true to replace an existing file's entire content with `content`. Leave false "
            "(the default) to create a new file only, failing if the path already exists. Use "
            "overwrite=true when you mean to rewrite a whole file in one step -- for changing "
            "part of a file, edit_file is still the right tool."
        ),
    )


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
        raise ToolError(
            "old_text must not be empty. edit_file replaces one exact existing block of text -- it "
            "cannot insert/append/rewrite a file with no anchor. If you haven't called read_file on "
            "this path yet in this conversation, do that first and choose a real block of its "
            "existing content as old_text. If you actually mean to replace the file's entire "
            "content, call write_file with overwrite=true and the complete new content instead."
        )
    if _looks_like_placeholder_token(args.old_text):
        raise ToolError(
            f"old_text ({args.old_text!r}) looks like an unfilled placeholder, not real content "
            f"copied from the file. Look at your most recent read_file('{args.path}') result in this "
            "conversation and copy an exact, real line (or block of lines) from it as old_text -- "
            "don't describe or paraphrase what you want to replace."
        )
    if _looks_like_placeholder_token(args.new_text):
        raise ToolError(
            f"new_text ({args.new_text!r}) looks like an unfilled placeholder, not real "
            "replacement content. Write out the actual code/text that should replace old_text."
        )
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

    new_normalized = _resolve_replacement(
        args.path, normalized, args.old_text, args.new_text, args.replace_all
    )
    if new_normalized == normalized:
        raise ValidationFailedError(
            f"The proposed edit would leave '{args.path}' unchanged; there's nothing to apply."
        )
    new_size = len(new_normalized.encode("utf-8"))
    if new_size > HARD_MAX_FILE_SIZE_BYTES:
        raise ToolError(f"The edited file would be {new_size:,} bytes, far too large.")
    _validate_python_syntax(args.path, new_normalized)

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
    if already_exists and not args.overwrite:
        raise ToolError(
            f"'{args.path}' already exists. Use edit_file to change part of it. If you actually "
            "mean to replace the file's entire content in one step -- for instance because "
            "edit_file keeps failing to match -- retry write_file with overwrite=true and the "
            "complete new content."
        )

    content_size = len(args.content.encode("utf-8"))
    if content_size > HARD_MAX_FILE_SIZE_BYTES:
        raise ToolError(f"The new file would be {content_size:,} bytes, far too large.")
    _validate_python_syntax(args.path, args.content)

    if already_exists:
        # Deliberately a whole-file replacement, not a patch, so there is no
        # read-before-write staleness check here: overwrite=true exists as the
        # escape hatch for when edit_file's anchor matching keeps failing, and
        # requiring a fresh read first would reintroduce the very failure it's
        # meant to route around. The user still sees the full diff against
        # what's on disk right now and still has to approve it -- that, not a
        # hash comparison, is what stops an unintended clobber. Modelled as
        # kind="edit" so the diff, the "Modified file" label, and the original
        # mode/newline preservation all come out right.
        try:
            existing_raw = target.read_bytes()
        except PermissionError as exc:
            raise PermissionDeniedError(
                f"Permission denied reading '{args.path}': {exc}"
            ) from exc
        old_normalized, newline = _decode_and_detect_newline(existing_raw)
        if old_normalized == args.content:
            raise ValidationFailedError(
                f"'{args.path}' already contains exactly this content; there's nothing to change."
            )
        change = ProposedChange(
            path=args.path,
            resolved_path=target,
            kind="edit",
            old_content=old_normalized,
            new_content=args.content,
            newline=newline,
            original_mode=target.stat().st_mode,
        )
        return ToolResult(
            ok=True,
            output="(pending user approval)",
            display=f"write_file({args.path!r}, overwrite=True)",
            pending_change=change,
        )

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
            "Propose replacing a block of text in an existing file with new text. Read the file "
            "first with read_file so old_text is its real, current content. IMPORTANT: read_file "
            "displays each line as 'NNN | code' — the line number, spaces and '|' are display "
            "only and are NOT in the file, so strip that prefix and pass just the code as "
            "old_text. Include enough surrounding context that old_text identifies exactly one "
            "place; if it appears more than once the tool tells you which lines, and you can "
            "either add context or pass replace_all=true to change every occurrence. To replace "
            "a whole file rather than a block, use write_file with overwrite=true. This only "
            "proposes the change: the user is shown a diff and must approve it before anything "
            "is written."
        ),
        args_model=EditFileArgs,
        run=lambda args: _edit_file(project, tracker, args),
    )


def build_write_file_tool(project: ProjectRoot, tracker: Optional[FileStateTracker] = None) -> Tool:
    return Tool(
        name="write_file",
        description=(
            "Propose creating a new file with the given content, or replacing an existing file's "
            "entire content when overwrite=true. Without overwrite it fails if the path already "
            "exists. Prefer edit_file for changing part of an existing file; reach for "
            "overwrite=true when you mean to rewrite the whole file, including when edit_file "
            "cannot match the text you want to change. This only proposes the change: the user "
            "is shown a diff and must approve it before anything is written."
        ),
        args_model=WriteFileArgs,
        run=lambda args: _write_file(project, tracker, args),
    )
