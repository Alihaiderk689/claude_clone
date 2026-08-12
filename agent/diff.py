"""Proposed file changes and unified diff generation.

This module is pure and presentation-agnostic — it only builds the data
and text for a proposed change. Terminal rendering (colors) lives in
cli.py, the only place that knows about Rich.
"""
from __future__ import annotations

import difflib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class ProposedChange:
    """A validated, not-yet-applied file modification.

    Carries everything needed to show the user a diff and, if approved,
    write the file — nothing is written to disk until a ProposedChange is
    handed to diff.apply_change() after explicit user approval.
    """

    path: str  # display path, relative to the project root
    resolved_path: Path  # absolute path inside the project root
    kind: str  # "edit" or "create"
    old_content: Optional[str]  # None for a new file; otherwise \n-normalized
    new_content: str  # \n-normalized
    newline: str = "\n"  # original line-ending style, restored on write
    original_mode: Optional[int] = None  # POSIX mode to preserve, if editing


def unified_diff_text(change: ProposedChange, context_lines: int = 3) -> str:
    old_lines = (change.old_content or "").splitlines(keepends=True)
    new_lines = change.new_content.splitlines(keepends=True)
    from_file = "/dev/null" if change.kind == "create" else f"a/{change.path}"
    to_file = f"b/{change.path}"
    diff = difflib.unified_diff(
        old_lines, new_lines, fromfile=from_file, tofile=to_file, n=context_lines
    )
    return "".join(diff)
