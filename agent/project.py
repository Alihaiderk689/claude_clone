"""Project-root awareness and shared path-safety rules.

Every filesystem-touching tool goes through ProjectRoot.resolve() so path
traversal and sensitive-file access are enforced in exactly one place,
never trusting arguments the model supplies.
"""
from __future__ import annotations

import fnmatch
from pathlib import Path
from typing import Iterable, Optional, Set

DEFAULT_IGNORED_DIRS: Set[str] = {
    ".git",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
    "env",
    "dist",
    "build",
    "coverage",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".idea",
    ".vscode",
    ".egg-info",
}

DEFAULT_SENSITIVE_PATTERNS = [
    "*.pem",
    "*.key",
    "id_rsa",
    "id_rsa.*",
    "*.pfx",
    "*.p12",
    "*credentials*.json",
    "secrets.*",
    "*.sqlite3",
]

# .env files are deliberately NOT in DEFAULT_SENSITIVE_PATTERNS -- by
# request, the agent is allowed to create/edit them directly (e.g. "make
# the API and put the secrets in .env"). They're kept here, checked
# separately, only for git.py's stage-time warning: accidentally
# `git add`-ing a real .env into a commit (and potentially a public
# remote) is a much harder mistake to undo than the agent merely being
# able to write the file locally, so that specific safety net stays on
# even though direct editing no longer requires it.
ENV_FILE_PATTERNS = [".env", ".env.*"]

# Kept small and CPU/RAM-friendly for an 8GB machine running a 7B model locally.
MAX_FILE_SIZE_BYTES = 200_000
# Even a line-range read materializes the whole file in memory today, so cap
# it well below anything that would meaningfully strain 8GB of RAM.
HARD_MAX_FILE_SIZE_BYTES = 20_000_000
MAX_FILE_LINES = 800
MAX_LIST_ENTRIES = 400
MAX_SEARCH_RESULTS = 50


class PathSecurityError(Exception):
    """Raised when a requested path escapes the project root or is blocked."""


class ProjectRoot:
    """Resolves user/model-supplied paths safely within a fixed project root."""

    def __init__(
        self,
        root: Path,
        ignored_dirs: Optional[Iterable[str]] = None,
        sensitive_patterns: Optional[Iterable[str]] = None,
    ) -> None:
        self.root = Path(root).resolve()
        self.ignored_dirs: Set[str] = (
            set(ignored_dirs) if ignored_dirs is not None else set(DEFAULT_IGNORED_DIRS)
        )
        self.sensitive_patterns = (
            list(sensitive_patterns) if sensitive_patterns is not None else list(DEFAULT_SENSITIVE_PATTERNS)
        )

    def resolve(self, user_path: str) -> Path:
        """Resolve a path (relative or absolute) and guarantee it stays inside root.

        Raises PathSecurityError for `..` traversal, absolute paths outside
        the root, or symlinks that resolve outside the root.
        """
        raw = user_path if user_path not in (None, "") else "."

        # A model that doesn't know the real path sometimes sends an unfilled
        # template token instead (observed live: read_file(path="<file_path>")
        # retried verbatim several times in a row). Angle brackets never
        # appear in a real project path on any platform this runs on, so this
        # is a reliable signal to reject early with a directive toward the
        # actual fix (list_files/search_files) instead of a generic "not
        # found" that invites retrying the same placeholder.
        if "<" in raw or ">" in raw:
            raise PathSecurityError(
                f"'{user_path}' looks like an unfilled placeholder, not a real path. Call "
                "list_files or search_files to find the actual path in this project, then retry "
                "with that real path -- do not guess or reuse a template token."
            )

        candidate = Path(raw)

        # A bare join of an absolute path onto root silently discards root
        # in pathlib, so absolute inputs must be resolved on their own and
        # then checked, never joined blindly.
        resolved = candidate.resolve() if candidate.is_absolute() else (self.root / candidate).resolve()

        try:
            resolved.relative_to(self.root)
        except ValueError:
            raise PathSecurityError(
                f"Access denied: '{user_path}' resolves outside the project root."
            )
        return resolved

    def is_ignored_dir(self, name: str) -> bool:
        return name in self.ignored_dirs

    def is_sensitive(self, path: Path) -> bool:
        name = path.name
        return any(fnmatch.fnmatch(name, pattern) for pattern in self.sensitive_patterns)

    def looks_like_env_file(self, path: Path) -> bool:
        """True for .env/.env.* regardless of whether .env is in this
        instance's sensitive_patterns -- used only by git.py's git_stage
        warning (see ENV_FILE_PATTERNS above), never by read/write/edit."""
        name = path.name
        return any(fnmatch.fnmatch(name, pattern) for pattern in ENV_FILE_PATTERNS)
