"""Validates Git operations before they're proposed to the user, and
carries the "already validated, not yet executed" data agent/tools/git.py
needs to show the user and (if approved) actually perform.

Read-only Git inspection (git_status/git_diff/git_log) doesn't need this --
it can't modify anything. This module guards only the three state-changing
operations: branch creation, staging, and committing. There is deliberately
no way to run an arbitrary Git command through here or anywhere else --
each operation gets its own narrow validator, the same way command_policy.py
allowlists specific programs instead of trusting an arbitrary string.
"""
from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from .project import PathSecurityError, ProjectRoot

MAX_COMMIT_MESSAGE_CHARS = 10_000
GIT_SUBPROCESS_TIMEOUT = 15

# Git ref names: must start with a letter/digit, then letters/digits/._-/ are
# fine. This is deliberately stricter than everything Git itself allows (no
# unicode, no leading dot) -- it's a first, cheap filter; git check-ref-format
# is the authoritative second check in validate_branch_name below.
_BRANCH_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]*$")

# Same rationale as command_policy.py: subprocess never uses shell=True, so
# none of these can actually trigger shell parsing. Checked anyway as
# defense in depth for branch names and paths, which Git itself parses
# structurally (unlike a commit message, which is just opaque text -- see
# validate_commit_message, which deliberately does NOT apply this check).
_SHELL_METACHARACTERS = (";", "&&", "||", "|", ">", "<", "$(", "`", "&", "\n", "\r", "\x00")


class GitPolicyError(Exception):
    """Raised with a model-facing explanation when a proposed Git operation is rejected."""


@dataclass(frozen=True)
class ProposedGitOperation:
    """A validated, not-yet-executed Git operation. agent/tools/git.py's
    apply_git_operation() is the only thing that ever acts on one, and only
    after the user has explicitly approved it.
    """

    kind: str  # "create_branch" | "stage" | "commit"
    repo_root: Path
    branch_name: Optional[str] = None
    paths: List[str] = field(default_factory=list)
    sensitive_paths: List[str] = field(default_factory=list)
    message: Optional[str] = None
    # Staged files at proposal time, for a commit -- re-checked at apply time
    # so a commit can't silently fire against a staging area that changed
    # underneath it (see Phase 5 spec item 22).
    expected_staged_files: List[str] = field(default_factory=list)


def _contains_shell_metacharacters(token: str) -> bool:
    return any(bad in token for bad in _SHELL_METACHARACTERS)


def validate_branch_name(repo_root: Path, name: str) -> str:
    name = (name or "").strip()
    if not name:
        raise GitPolicyError("Branch name must not be empty.")
    if name.startswith("-"):
        raise GitPolicyError(
            f"Branch name {name!r} must not start with '-' (Git could parse it as a flag)."
        )
    if _contains_shell_metacharacters(name):
        raise GitPolicyError(f"Branch name {name!r} contains disallowed characters.")
    if (
        not _BRANCH_NAME_RE.match(name)
        or ".." in name
        or "//" in name
        or name.endswith((".lock", ".", "/"))
    ):
        raise GitPolicyError(
            f"Branch name {name!r} is not a valid Git ref name (use letters, numbers, '.', '_', "
            "'-', '/', starting with a letter or number; no '..', and no trailing '.' or '/')."
        )

    try:
        result = subprocess.run(
            ["git", "check-ref-format", "--branch", name],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=GIT_SUBPROCESS_TIMEOUT,
        )
    except OSError as exc:
        raise GitPolicyError(f"Could not validate branch name: {exc}") from exc
    if result.returncode != 0:
        raise GitPolicyError(f"Branch name {name!r} is not a valid Git ref name.")

    return name


def validate_stage_paths(project: ProjectRoot, paths: List[str]) -> List[str]:
    """Resolve and validate each path through the same ProjectRoot used by
    every other tool -- traversal and absolute-path-outside-root are
    rejected exactly as they are for read_file/edit_file/write_file.
    Returns paths relative to the project root, suitable for `git add --`.
    """
    if not paths:
        raise GitPolicyError("At least one path must be provided to stage.")

    relative_paths: List[str] = []
    for raw in paths:
        if not raw or raw.startswith("-"):
            raise GitPolicyError(f"Invalid path {raw!r}.")
        if _contains_shell_metacharacters(raw):
            raise GitPolicyError(f"Path {raw!r} contains disallowed characters.")
        try:
            resolved = project.resolve(raw)
        except PathSecurityError as exc:
            raise GitPolicyError(str(exc)) from exc
        relative_paths.append(str(resolved.relative_to(project.root)))

    return relative_paths


def validate_commit_message(message: str) -> str:
    message = (message or "").strip()
    if not message:
        raise GitPolicyError("Commit message must not be empty.")
    if len(message) > MAX_COMMIT_MESSAGE_CHARS:
        raise GitPolicyError(f"Commit message is too long (max {MAX_COMMIT_MESSAGE_CHARS} characters).")
    if "\x00" in message:
        raise GitPolicyError("Commit message contains invalid characters.")
    return message
