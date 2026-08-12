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
    ".env",
    ".env.*",
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
