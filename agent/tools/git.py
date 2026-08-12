"""Git awareness and controlled Git operations.

Read-only inspection (git_status, git_diff, git_log) works like any other
read-only tool -- no approval needed since nothing is modified. The three
state-changing tools (git_create_branch, git_stage, git_commit) follow the
exact same propose/apply split as editing.py and terminal.py: run() only
validates (via ../git_policy.py) and builds a ProposedGitOperation
describing exactly what would happen; apply_git_operation() is the only
thing that ever actually runs `git branch`, `git add`, or `git commit`, and
it's only ever called by the agent loop after the user has explicitly
approved.

There is deliberately no generic "run a git command" tool. Each operation
is its own narrow, validated tool -- this is what keeps destructive Git
commands (reset --hard, clean -fd, checkout/restore ., push --force, branch
-D, rebase, merge, ...) permanently unreachable, not just discouraged.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple

from pydantic import BaseModel, Field

from ..git_policy import (
    GitPolicyError,
    ProposedGitOperation,
    validate_branch_name,
    validate_commit_message,
    validate_stage_paths,
)
from ..project import PathSecurityError, ProjectRoot
from .base import Tool, ToolError, ToolResult

MAX_LOG_ENTRIES = 20
MAX_DIFF_CHARS = 8000
GIT_SUBPROCESS_TIMEOUT = 15

# Same rationale as terminal.py's _minimal_env: never hand a subprocess the
# full parent environment, since incidental output (or a hook script) could
# echo secrets that would then land in what the model sees.
_SAFE_ENV_VARS = ("PATH", "HOME", "LANG", "LC_ALL", "LC_CTYPE", "TMPDIR", "SHELL")


class GitStatusArgs(BaseModel):
    pass


class GitDiffArgs(BaseModel):
    staged: bool = Field(
        default=False,
        description="Show the staged (index) diff instead of the unstaged working-tree diff.",
    )
    path: Optional[str] = Field(
        default=None, description="Limit the diff to one file/directory, relative to the project root."
    )


class GitLogArgs(BaseModel):
    limit: int = Field(
        default=10, ge=1, le=MAX_LOG_ENTRIES, description=f"Number of recent commits (max {MAX_LOG_ENTRIES})."
    )


class GitCreateBranchArgs(BaseModel):
    name: str = Field(description="New branch name to create, e.g. 'fix/login-authentication'.")


class GitStageArgs(BaseModel):
    paths: List[str] = Field(description="Files to stage, relative to the project root.")


class GitCommitArgs(BaseModel):
    message: str = Field(description="Commit message.")


def _minimal_env() -> dict:
    return {name: os.environ[name] for name in _SAFE_ENV_VARS if name in os.environ}


def _run_git(repo_root: Path, args: List[str], timeout: int = GIT_SUBPROCESS_TIMEOUT) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        timeout=timeout,
        env=_minimal_env(),
    )


def is_git_repository(repo_root: Path) -> bool:
    try:
        result = _run_git(repo_root, ["rev-parse", "--is-inside-work-tree"])
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0 and result.stdout.strip() == "true"


def _require_git_repo(repo_root: Path) -> None:
    if not is_git_repository(repo_root):
        raise ToolError("This project is not a Git repository.")


def _current_branch(repo_root: Path) -> Tuple[Optional[str], bool, Optional[str]]:
    """Returns (branch_name_or_None, is_detached, short_sha_if_detached)."""
    result = _run_git(repo_root, ["branch", "--show-current"])
    name = result.stdout.strip()
    if name:
        return name, False, None
    sha_result = _run_git(repo_root, ["rev-parse", "--short", "HEAD"])
    return None, True, (sha_result.stdout.strip() or None)


def _parse_status_line(line: str):
    # NOTE: filenames containing spaces or unusual characters are shown
    # quoted by `git status --porcelain=v1` (without -z); this parser does
    # not unquote them, so such paths pass through with literal quotes. A
    # known, acceptable limitation for a first Git-awareness pass.
    x, y = line[0], line[1]
    rest = line[3:]
    if " -> " in rest:
        orig, _, path = rest.partition(" -> ")
        return x, y, path, orig
    return x, y, rest, None


def _git_status(project: ProjectRoot, _args: GitStatusArgs) -> ToolResult:
    _require_git_repo(project.root)

    branch, detached, sha = _current_branch(project.root)
    result = _run_git(project.root, ["status", "--porcelain=v1"])
    if result.returncode != 0:
        raise ToolError(f"git status failed: {result.stderr.strip()[:300]}")

    staged: List[str] = []
    modified: List[str] = []
    untracked: List[str] = []
    deleted: List[str] = []
    renamed: List[str] = []

    for line in result.stdout.splitlines():
        if not line:
            continue
        x, y, path, orig = _parse_status_line(line)
        if x == "?" and y == "?":
            untracked.append(path)
            continue
        if orig:
            renamed.append(f"{orig} -> {path}")
        elif x != " ":
            staged.append(f"{path} (deleted)" if x == "D" else path)
        if y == "M":
            modified.append(path)
        elif y == "D":
            deleted.append(path)

    lines = []
    if detached:
        lines.append(f"Branch: (detached HEAD at {sha})" if sha else "Branch: (detached HEAD)")
    else:
        lines.append(f"Branch: {branch}")

    if not (staged or modified or untracked or deleted or renamed):
        lines.append("\nWorking tree: clean")
    else:
        for label, items in (
            ("Staged", staged),
            ("Modified", modified),
            ("Deleted", deleted),
            ("Renamed", renamed),
            ("Untracked", untracked),
        ):
            if items:
                lines.append(f"\n{label}:")
                lines.extend(f"  {p}" for p in items)

    return ToolResult(ok=True, output="\n".join(lines), display="git_status()")


def _git_diff(project: ProjectRoot, args: GitDiffArgs) -> ToolResult:
    _require_git_repo(project.root)

    git_args = ["diff"]
    if args.staged:
        git_args.append("--staged")
    if args.path:
        try:
            resolved = project.resolve(args.path)
        except PathSecurityError as exc:
            raise ToolError(str(exc)) from exc
        git_args += ["--", str(resolved.relative_to(project.root))]

    result = _run_git(project.root, git_args)
    display = f"git_diff(staged={args.staged})"
    if result.returncode != 0:
        raise ToolError(f"git diff failed: {result.stderr.strip()[:300]}")

    diff_text = result.stdout
    if not diff_text.strip():
        label = "staged" if args.staged else "unstaged"
        return ToolResult(ok=True, output=f"No {label} changes.", display=display)

    if len(diff_text) > MAX_DIFF_CHARS:
        diff_text = diff_text[:MAX_DIFF_CHARS] + (
            f"\n[diff truncated because it exceeded the {MAX_DIFF_CHARS}-character limit]"
        )

    return ToolResult(ok=True, output=diff_text, display=display)


def _git_log(project: ProjectRoot, args: GitLogArgs) -> ToolResult:
    _require_git_repo(project.root)

    result = _run_git(project.root, ["log", f"-n{args.limit}", "--pretty=format:%h %s"])
    display = f"git_log(limit={args.limit})"
    if result.returncode != 0:
        stderr = result.stderr.lower()
        if "does not have any commits yet" in stderr or "unknown revision" in stderr:
            return ToolResult(ok=True, output="This repository has no commits yet.", display=display)
        raise ToolError(f"git log failed: {result.stderr.strip()[:300]}")

    output = result.stdout.strip()
    if not output:
        return ToolResult(ok=True, output="This repository has no commits yet.", display=display)

    return ToolResult(ok=True, output=output, display=display)


def _git_create_branch(project: ProjectRoot, args: GitCreateBranchArgs) -> ToolResult:
    _require_git_repo(project.root)
    try:
        name = validate_branch_name(project.root, args.name)
    except GitPolicyError as exc:
        raise ToolError(str(exc)) from exc

    existing = _run_git(project.root, ["branch", "--list", name])
    if existing.stdout.strip():
        raise ToolError(f"Branch {name!r} already exists.")

    op = ProposedGitOperation(kind="create_branch", repo_root=project.root, branch_name=name)
    return ToolResult(
        ok=True,
        output="(pending user approval)",
        display=f"git_create_branch({name!r})",
        pending_git_operation=op,
    )


def _git_stage(project: ProjectRoot, args: GitStageArgs) -> ToolResult:
    _require_git_repo(project.root)
    try:
        relative_paths = validate_stage_paths(project, args.paths)
    except GitPolicyError as exc:
        raise ToolError(str(exc)) from exc

    sensitive = [p for p in relative_paths if project.is_sensitive(project.root / p)]

    op = ProposedGitOperation(
        kind="stage", repo_root=project.root, paths=relative_paths, sensitive_paths=sensitive
    )
    return ToolResult(
        ok=True,
        output="(pending user approval)",
        display=f"git_stage({relative_paths!r})",
        pending_git_operation=op,
    )


def _git_commit(project: ProjectRoot, args: GitCommitArgs) -> ToolResult:
    _require_git_repo(project.root)
    try:
        message = validate_commit_message(args.message)
    except GitPolicyError as exc:
        raise ToolError(str(exc)) from exc

    staged_result = _run_git(project.root, ["diff", "--staged", "--name-only"])
    staged_files = [line for line in staged_result.stdout.splitlines() if line]
    if not staged_files:
        raise ToolError("Nothing is staged. Use git_stage to stage files before proposing a commit.")

    op = ProposedGitOperation(
        kind="commit", repo_root=project.root, message=message, expected_staged_files=staged_files
    )
    return ToolResult(
        ok=True,
        output="(pending user approval)",
        display="git_commit(...)",
        pending_git_operation=op,
    )


def apply_git_operation(op: ProposedGitOperation) -> ToolResult:
    """Actually perform an approved Git operation. Only ever called by the
    agent loop after the user has explicitly approved -- never by a tool's
    run(). This is the only function in the codebase that ever runs
    `git branch`, `git add`, or `git commit`.
    """
    if not is_git_repository(op.repo_root):
        return ToolResult(
            ok=False, output="This project is not a Git repository anymore. Nothing was done."
        )

    if op.kind == "create_branch":
        result = _run_git(op.repo_root, ["branch", "--", op.branch_name])
        if result.returncode != 0:
            return ToolResult(ok=False, output=f"Failed to create branch: {result.stderr.strip()[:300]}")
        return ToolResult(
            ok=True, output=f"Created branch '{op.branch_name}'. It has not been checked out."
        )

    if op.kind == "stage":
        result = _run_git(op.repo_root, ["add", "--", *op.paths])
        if result.returncode != 0:
            return ToolResult(ok=False, output=f"Failed to stage files: {result.stderr.strip()[:300]}")
        return ToolResult(ok=True, output=f"Staged {len(op.paths)} file(s): {', '.join(op.paths)}")

    if op.kind == "commit":
        current = _run_git(op.repo_root, ["diff", "--staged", "--name-only"])
        current_files = sorted(line for line in current.stdout.splitlines() if line)
        if current_files != sorted(op.expected_staged_files):
            return ToolResult(
                ok=False,
                output=(
                    "The staged files changed since this commit was proposed (expected "
                    f"{sorted(op.expected_staged_files)}, found {current_files}). Refusing to "
                    "commit a stale proposal -- run git_status again and re-propose."
                ),
            )
        result = _run_git(op.repo_root, ["commit", "-m", op.message])
        if result.returncode != 0:
            return ToolResult(ok=False, output=f"Failed to commit: {result.stderr.strip()[:300]}")
        sha_result = _run_git(op.repo_root, ["rev-parse", "--short", "HEAD"])
        sha = sha_result.stdout.strip()
        return ToolResult(ok=True, output=f"Created commit {sha}: {op.message}")

    return ToolResult(ok=False, output=f"Unknown Git operation kind: {op.kind!r}")  # pragma: no cover


def build_git_status_tool(project: ProjectRoot) -> Tool:
    return Tool(
        name="git_status",
        description=(
            "Show the current Git branch (or detached-HEAD state) and a structured summary of "
            "staged, modified, deleted, renamed, and untracked files. Read-only; safe to call "
            "any time. Reports clearly if this project isn't a Git repository instead of failing."
        ),
        args_model=GitStatusArgs,
        run=lambda args: _git_status(project, args),
    )


def build_git_diff_tool(project: ProjectRoot) -> Tool:
    return Tool(
        name="git_diff",
        description=(
            "Show a unified diff of changes: staged=false (default) for unstaged working-tree "
            "changes, staged=true for what's staged for the next commit. Optionally limit to one "
            "file/directory with `path`. Read-only. Large diffs are truncated with a clear marker."
        ),
        args_model=GitDiffArgs,
        run=lambda args: _git_diff(project, args),
    )


def build_git_log_tool(project: ProjectRoot) -> Tool:
    return Tool(
        name="git_log",
        description=(
            f"Show up to {MAX_LOG_ENTRIES} recent commits as '<short-hash> <subject>' lines, "
            "most recent first. Read-only."
        ),
        args_model=GitLogArgs,
        run=lambda args: _git_log(project, args),
    )


def build_git_create_branch_tool(project: ProjectRoot) -> Tool:
    return Tool(
        name="git_create_branch",
        description=(
            "Propose creating a new Git branch from the current HEAD. Only proposes it -- the "
            "user is shown the branch name and must approve before it's created. The new branch "
            "is not checked out. Fails if the name is invalid or already exists."
        ),
        args_model=GitCreateBranchArgs,
        run=lambda args: _git_create_branch(project, args),
    )


def build_git_stage_tool(project: ProjectRoot) -> Tool:
    return Tool(
        name="git_stage",
        description=(
            "Propose staging one or more files for the next commit (like `git add`). Only "
            "proposes it -- the user is shown the exact file list and must approve before "
            "anything is staged. Files matching sensitive patterns (.env, *.pem, *.key, "
            "credentials.*, etc.) trigger an explicit warning the user must acknowledge."
        ),
        args_model=GitStageArgs,
        run=lambda args: _git_stage(project, args),
    )


def build_git_commit_tool(project: ProjectRoot) -> Tool:
    return Tool(
        name="git_commit",
        description=(
            "Propose creating a commit from the currently staged files with the given message. "
            "Only proposes it -- the user is shown the message and the staged file list and must "
            "approve before anything is committed. Fails if nothing is staged. Inspect the staged "
            "diff with git_diff(staged=true) before proposing a commit, and stage files with "
            "git_stage first if needed."
        ),
        args_model=GitCommitArgs,
        run=lambda args: _git_commit(project, args),
    )
