"""Tracks the content the model has actually seen, to catch stale edits.

When read_file is called, we remember a hash of the file's full content.
edit_file checks that hash against the file's current content before
proposing a change, so an edit built from a stale read can't silently
overwrite content the model never actually saw. If a path was never read,
there's nothing to compare against, so the check simply doesn't apply.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict


class FileStateTracker:
    def __init__(self) -> None:
        self._hashes: Dict[Path, str] = {}

    def record(self, path: Path, content: str) -> None:
        self._hashes[path] = self._hash(content)

    def is_fresh(self, path: Path, content: str) -> bool:
        """True if `content` matches what was last recorded for `path`, or
        if nothing has been recorded for it yet (nothing to compare)."""
        recorded = self._hashes.get(path)
        return recorded is None or recorded == self._hash(content)

    def forget(self, path: Path) -> None:
        self._hashes.pop(path, None)

    @staticmethod
    def _hash(content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()
