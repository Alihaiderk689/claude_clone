"""Validates delete/rename filesystem mutations before they're proposed to
the user, and carries the "already validated, not yet executed" data
agent/tools/file_ops.py needs to show the user and (if approved) actually
perform.

Mirrors git_policy.py's split from agent/tools/git.py: this module only
validates (path safety, existence, collisions) and never touches the
filesystem; agent/tools/file_ops.py's apply_file_op() is the only thing that
ever actually calls os.remove/os.rename, and only after explicit user
approval. Deliberately scoped to files only, not directories -- a recursive
directory delete/move is a meaningfully larger blast radius than anything
else this agent can do, and the sorting.py-style requests this was built for
(delete/rename one obsolete file) never need it.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .project import PathSecurityError, ProjectRoot


class FileOpError(Exception):
    """Raised with a model-facing explanation when a proposed delete/rename is rejected."""


@dataclass(frozen=True)
class ProposedFileOp:
    """A validated, not-yet-executed delete or rename/move.

    agent/tools/file_ops.py's apply_file_op() is the only thing that ever
    acts on one, and only after the user has explicitly approved it.
    """

    kind: str  # "delete" | "rename"
    source_path: str  # display path, relative to the project root
    resolved_source: Path  # absolute path inside the project root
    destination_path: Optional[str] = None  # rename only
    resolved_destination: Optional[Path] = None  # rename only


def _resolve_in_project(project: ProjectRoot, path: str) -> Path:
    try:
        return project.resolve(path)
    except PathSecurityError as exc:
        raise FileOpError(str(exc)) from exc


def _reject_project_root(project: ProjectRoot, resolved: Path, label: str) -> None:
    if resolved == project.root:
        raise FileOpError(f"Refusing to {label} the project root itself.")


def validate_delete(project: ProjectRoot, path: str) -> ProposedFileOp:
    if not path or path in (".", "/"):
        raise FileOpError("A specific file path is required to delete.")

    resolved = _resolve_in_project(project, path)
    _reject_project_root(project, resolved, "delete")

    if not resolved.exists():
        raise FileOpError(f"File not found: '{path}'.")
    if resolved.is_dir():
        raise FileOpError(
            f"'{path}' is a directory. delete_file only removes a single file, "
            "not a directory or its contents."
        )
    if project.is_sensitive(resolved):
        raise FileOpError(f"Refusing to delete '{path}': it matches a sensitive-file pattern.")

    return ProposedFileOp(kind="delete", source_path=path, resolved_source=resolved)


def validate_rename(project: ProjectRoot, source: str, destination: str) -> ProposedFileOp:
    if not source or not destination:
        raise FileOpError("Both source and destination paths are required.")
    if source in (".", "/") or destination in (".", "/"):
        raise FileOpError("Source and destination must be specific file paths.")

    resolved_source = _resolve_in_project(project, source)
    resolved_destination = _resolve_in_project(project, destination)
    _reject_project_root(project, resolved_source, "rename")
    _reject_project_root(project, resolved_destination, "rename")

    if resolved_source == resolved_destination:
        raise FileOpError("Source and destination are the same path; nothing to rename.")
    if not resolved_source.exists():
        raise FileOpError(f"File not found: '{source}'.")
    if resolved_source.is_dir():
        raise FileOpError(
            f"'{source}' is a directory. rename_file only renames/moves a single file, "
            "not a directory or its contents."
        )
    if resolved_destination.exists():
        raise FileOpError(
            f"'{destination}' already exists. rename_file refuses to overwrite an existing file -- "
            f"delete_file it first if you actually intend to replace it, then retry the rename."
        )
    if project.is_sensitive(resolved_source):
        raise FileOpError(f"Refusing to rename '{source}': it matches a sensitive-file pattern.")
    if project.is_sensitive(resolved_destination):
        raise FileOpError(
            f"Refusing to rename to '{destination}': it matches a sensitive-file pattern."
        )

    return ProposedFileOp(
        kind="rename",
        source_path=source,
        resolved_source=resolved_source,
        destination_path=destination,
        resolved_destination=resolved_destination,
    )
