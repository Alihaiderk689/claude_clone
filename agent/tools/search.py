"""Code search tool: search_files.

Prefers the system `rg` (ripgrep) binary when available since it is fast
and already knows how to skip binary files; falls back to a small pure
-Python substring scan otherwise. Both paths respect the same ignored
-directory and sensitive-file rules as the other tools.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import List

from pydantic import BaseModel, Field

from ..project import MAX_SEARCH_RESULTS, PathSecurityError, ProjectRoot
from .base import Tool, ToolError, ToolResult

_PER_FILE_MATCH_CAP = 5
_MAX_SCAN_FILE_BYTES = 1_000_000
_SEARCH_TIMEOUT_SECONDS = 10
_LINE_PREVIEW_CHARS = 200


class SearchFilesArgs(BaseModel):
    query: str = Field(description="Literal text to search for in the codebase (case-sensitive).")
    path: str = Field(
        default=".",
        description="Directory to search within, relative to the project root.",
    )
    max_results: int = Field(
        default=30,
        ge=1,
        le=MAX_SEARCH_RESULTS,
        description="Maximum number of matching lines to return.",
    )


def _search_files(project: ProjectRoot, args: SearchFilesArgs) -> ToolResult:
    if not args.query.strip():
        raise ToolError("query must not be empty.")

    try:
        target = project.resolve(args.path)
    except PathSecurityError as exc:
        raise ToolError(str(exc)) from exc

    if not target.exists():
        raise ToolError(f"Path not found: '{args.path}'")

    if shutil.which("rg"):
        results = _search_with_ripgrep(project, target, args)
    else:
        results = _search_with_fallback(project, target, args)

    if not results:
        return ToolResult(
            ok=True,
            output=f"No matches for '{args.query}'.",
            display=f"search_files({args.query!r})",
        )

    output = "\n".join(results)
    if len(results) >= args.max_results:
        output += f"\n... results capped at {args.max_results}. Narrow the query or path for more."

    return ToolResult(ok=True, output=output, display=f"search_files({args.query!r})")


def _search_with_ripgrep(project: ProjectRoot, target: Path, args: SearchFilesArgs) -> List[str]:
    cmd = [
        "rg",
        "--line-number",
        "--no-heading",
        "--color=never",
        "-m",
        str(_PER_FILE_MATCH_CAP),
    ]
    for pattern in sorted(project.ignored_dirs):
        cmd += ["--glob", f"!**/{pattern}/**"]
    for pattern in project.sensitive_patterns:
        cmd += ["--glob", f"!{pattern}"]
    cmd += ["--", args.query, str(target)]

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=_SEARCH_TIMEOUT_SECONDS,
            cwd=str(project.root),
        )
    except subprocess.TimeoutExpired:
        raise ToolError("Search timed out.")
    except OSError as exc:
        raise ToolError(f"Failed to run ripgrep: {exc}")

    # rg exits 1 when there are simply no matches; only >1 is a real error.
    if proc.returncode not in (0, 1):
        raise ToolError(f"ripgrep error: {proc.stderr.strip()[:300]}")

    results: List[str] = []
    for line in proc.stdout.splitlines():
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        path_part, line_no, content = parts
        try:
            rel = Path(path_part).resolve().relative_to(project.root)
        except ValueError:
            continue
        results.append(f"{rel}:{line_no}: {content.strip()[:_LINE_PREVIEW_CHARS]}")
        if len(results) >= args.max_results:
            break
    return results


def _iter_project_files(project: ProjectRoot, target: Path):
    if target.is_file():
        yield target
        return
    for current_root, dirnames, filenames in os.walk(target):
        dirnames[:] = [d for d in dirnames if not project.is_ignored_dir(d)]
        for name in filenames:
            path = Path(current_root) / name
            if project.is_sensitive(path):
                continue
            yield path


def _search_with_fallback(project: ProjectRoot, target: Path, args: SearchFilesArgs) -> List[str]:
    results: List[str] = []
    for path in _iter_project_files(project, target):
        if len(results) >= args.max_results:
            break
        try:
            if path.stat().st_size > _MAX_SCAN_FILE_BYTES:
                continue
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue

        matches_in_file = 0
        for i, line in enumerate(text.splitlines(), start=1):
            if args.query in line:
                rel = path.relative_to(project.root)
                results.append(f"{rel}:{i}: {line.strip()[:_LINE_PREVIEW_CHARS]}")
                matches_in_file += 1
                if matches_in_file >= _PER_FILE_MATCH_CAP or len(results) >= args.max_results:
                    break
    return results


def build_search_files_tool(project: ProjectRoot) -> Tool:
    return Tool(
        name="search_files",
        description=(
            "Search the project for a literal text string (case-sensitive), returning matching "
            "file paths, line numbers, and line content. Uses ripgrep when available. Prefer this "
            "over read_file when you don't yet know which file contains what you're looking for."
        ),
        args_model=SearchFilesArgs,
        run=lambda args: _search_files(project, args),
    )
