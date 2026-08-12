"""Tests for the Git tools in agent/tools/git.py.

Uses real temporary Git repositories (git is fast, safe, and read-only for
most of what's tested here) rather than mocking subprocess -- this is the
only way to be confident the porcelain-output parsing is actually correct.
The few failure paths that are impractical to trigger for real (missing
git binary, OS-level errors) are covered with subprocess mocked instead.
"""
from __future__ import annotations

import subprocess
from unittest import mock

import pytest

from agent.project import ProjectRoot
from agent.tools.git import (
    apply_git_operation,
    build_git_commit_tool,
    build_git_create_branch_tool,
    build_git_diff_tool,
    build_git_log_tool,
    build_git_stage_tool,
    build_git_status_tool,
    is_git_repository,
)


def run_git(repo_root, *args):
    subprocess.run(["git", *args], cwd=str(repo_root), check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path):
    run_git(tmp_path, "init", "-q")
    run_git(tmp_path, "config", "user.email", "test@example.com")
    run_git(tmp_path, "config", "user.name", "Test")
    (tmp_path / "README.md").write_text("# Test repo\n")
    (tmp_path / "backend").mkdir()
    (tmp_path / "backend" / "auth.py").write_text("def login():\n    pass\n")
    run_git(tmp_path, "add", "README.md", "backend/auth.py")
    run_git(tmp_path, "commit", "-q", "-m", "Initial commit")
    return tmp_path


@pytest.fixture
def project(repo):
    return ProjectRoot(repo)


@pytest.fixture
def non_git_project(tmp_path):
    (tmp_path / "some_file.py").write_text("x = 1\n")
    return ProjectRoot(tmp_path)


class TestIsGitRepository:
    def test_true_for_real_repo(self, repo):
        assert is_git_repository(repo) is True

    def test_false_for_non_repo(self, tmp_path):
        assert is_git_repository(tmp_path) is False

    def test_false_when_git_binary_missing(self, tmp_path):
        with mock.patch("agent.tools.git.subprocess.run", side_effect=FileNotFoundError()):
            assert is_git_repository(tmp_path) is False


class TestGitStatusNonRepo:
    def test_reports_not_a_repo_without_crashing(self, non_git_project):
        tool = build_git_status_tool(non_git_project)
        result = tool.execute({})
        assert not result.ok
        assert result.output == "This project is not a Git repository."


class TestGitStatus:
    def test_clean_repository(self, project):
        tool = build_git_status_tool(project)
        result = tool.execute({})
        assert result.ok
        assert "Branch:" in result.output
        assert "Working tree: clean" in result.output

    def test_reports_current_branch(self, project, repo):
        branch = subprocess.run(
            ["git", "branch", "--show-current"], cwd=str(repo), capture_output=True, text=True
        ).stdout.strip()
        tool = build_git_status_tool(project)
        result = tool.execute({})
        assert f"Branch: {branch}" in result.output

    def test_modified_file(self, project, repo):
        (repo / "README.md").write_text("changed\n")
        result = build_git_status_tool(project).execute({})
        assert "Modified:" in result.output
        assert "README.md" in result.output

    def test_untracked_file(self, project, repo):
        (repo / "new_file.py").write_text("x = 1\n")
        result = build_git_status_tool(project).execute({})
        assert "Untracked:" in result.output
        assert "new_file.py" in result.output

    def test_staged_file(self, project, repo):
        (repo / "README.md").write_text("changed\n")
        run_git(repo, "add", "README.md")
        result = build_git_status_tool(project).execute({})
        assert "Staged:" in result.output
        assert "README.md" in result.output

    def test_deleted_file_unstaged(self, project, repo):
        (repo / "README.md").unlink()
        result = build_git_status_tool(project).execute({})
        assert "Deleted:" in result.output
        assert "README.md" in result.output

    def test_deleted_file_staged(self, project, repo):
        (repo / "README.md").unlink()
        run_git(repo, "add", "README.md")
        result = build_git_status_tool(project).execute({})
        assert "Staged:" in result.output
        assert "README.md (deleted)" in result.output

    def test_detached_head(self, project, repo):
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(repo), capture_output=True, text=True
        ).stdout.strip()
        run_git(repo, "checkout", "-q", sha)
        result = build_git_status_tool(project).execute({})
        assert result.ok
        assert "detached HEAD" in result.output


class TestGitDiff:
    def test_no_unstaged_changes(self, project):
        result = build_git_diff_tool(project).execute({})
        assert result.ok
        assert "No unstaged changes." in result.output

    def test_unstaged_diff_shows_change(self, project, repo):
        (repo / "README.md").write_text("changed content\n")
        result = build_git_diff_tool(project).execute({"staged": False})
        assert "-# Test repo" in result.output
        assert "+changed content" in result.output

    def test_staged_diff(self, project, repo):
        (repo / "README.md").write_text("changed content\n")
        run_git(repo, "add", "README.md")
        unstaged = build_git_diff_tool(project).execute({"staged": False})
        staged = build_git_diff_tool(project).execute({"staged": True})
        assert "No unstaged changes." in unstaged.output
        assert "+changed content" in staged.output

    def test_no_staged_changes_message(self, project):
        result = build_git_diff_tool(project).execute({"staged": True})
        assert "No staged changes." in result.output

    def test_diff_limited_to_path(self, project, repo):
        (repo / "README.md").write_text("changed\n")
        (repo / "backend" / "auth.py").write_text("def login():\n    return True\n")
        result = build_git_diff_tool(project).execute({"path": "README.md"})
        assert "README.md" in result.output
        assert "auth.py" not in result.output

    def test_diff_path_traversal_rejected(self, project):
        result = build_git_diff_tool(project).execute({"path": "../../etc/passwd"})
        assert not result.ok
        assert "access denied" in result.output.lower()

    def test_large_diff_is_truncated(self, project, repo):
        big_content = "\n".join(f"line {i}" for i in range(5000))
        (repo / "backend" / "auth.py").write_text(big_content)
        result = build_git_diff_tool(project).execute({})
        assert result.ok
        assert "truncated" in result.output.lower()

    def test_non_repo_reported_gracefully(self, non_git_project):
        result = build_git_diff_tool(non_git_project).execute({})
        assert not result.ok
        assert result.output == "This project is not a Git repository."


class TestGitLog:
    def test_recent_commits(self, project, repo):
        (repo / "README.md").write_text("v2\n")
        run_git(repo, "commit", "-q", "-a", "-m", "Second commit")
        result = build_git_log_tool(project).execute({})
        assert result.ok
        lines = result.output.splitlines()
        assert len(lines) == 2
        assert "Second commit" in lines[0]
        assert "Initial commit" in lines[1]

    def test_limit_enforced(self, project, repo):
        for i in range(5):
            (repo / "README.md").write_text(f"v{i}\n")
            run_git(repo, "commit", "-q", "-a", "-m", f"commit {i}")
        result = build_git_log_tool(project).execute({"limit": 3})
        assert len(result.output.splitlines()) == 3

    def test_limit_above_max_rejected(self, project):
        result = build_git_log_tool(project).execute({"limit": 999})
        assert not result.ok

    def test_empty_history(self, non_git_project, tmp_path):
        run_git(tmp_path, "init", "-q")
        run_git(tmp_path, "config", "user.email", "test@example.com")
        run_git(tmp_path, "config", "user.name", "Test")
        project = ProjectRoot(tmp_path)
        result = build_git_log_tool(project).execute({})
        assert result.ok
        assert "no commits yet" in result.output.lower()

    def test_short_hash_format(self, project):
        result = build_git_log_tool(project).execute({})
        first_token = result.output.split()[0]
        assert 7 <= len(first_token) <= 12
        assert all(c in "0123456789abcdef" for c in first_token)


class TestGitCreateBranchPropose:
    def test_valid_branch_proposed(self, project):
        result = build_git_create_branch_tool(project).execute({"name": "feature/x"})
        assert result.ok
        assert result.pending_git_operation is not None
        assert result.pending_git_operation.kind == "create_branch"
        assert result.pending_git_operation.branch_name == "feature/x"

    def test_invalid_name_rejected(self, project):
        result = build_git_create_branch_tool(project).execute({"name": "-bad"})
        assert not result.ok
        assert result.pending_git_operation is None

    def test_already_existing_branch_rejected(self, project, repo):
        run_git(repo, "branch", "existing-branch")
        result = build_git_create_branch_tool(project).execute({"name": "existing-branch"})
        assert not result.ok
        assert "already exists" in result.output.lower()

    def test_non_repo_reported_gracefully(self, non_git_project):
        result = build_git_create_branch_tool(non_git_project).execute({"name": "feature/x"})
        assert not result.ok
        assert result.output == "This project is not a Git repository."


class TestGitStagePropose:
    def test_valid_paths_proposed(self, project):
        result = build_git_stage_tool(project).execute({"paths": ["backend/auth.py"]})
        assert result.ok
        assert result.pending_git_operation.paths == ["backend/auth.py"]
        assert result.pending_git_operation.sensitive_paths == []

    def test_path_traversal_rejected(self, project):
        result = build_git_stage_tool(project).execute({"paths": ["../../etc/passwd"]})
        assert not result.ok

    def test_sensitive_file_flagged_not_silently_dropped(self, project, repo):
        (repo / ".env").write_text("SECRET=1\n")
        result = build_git_stage_tool(project).execute({"paths": [".env"]})
        assert result.ok  # still proposed -- flagged, not silently rejected
        assert ".env" in result.pending_git_operation.sensitive_paths

    def test_mixed_sensitive_and_normal_paths(self, project, repo):
        (repo / ".env").write_text("SECRET=1\n")
        result = build_git_stage_tool(project).execute({"paths": ["backend/auth.py", ".env"]})
        assert result.ok
        op = result.pending_git_operation
        assert set(op.paths) == {"backend/auth.py", ".env"}
        assert op.sensitive_paths == [".env"]

    def test_empty_paths_rejected(self, project):
        result = build_git_stage_tool(project).execute({"paths": []})
        assert not result.ok


class TestGitCommitPropose:
    def test_nothing_staged_rejected(self, project):
        result = build_git_commit_tool(project).execute({"message": "Fix bug"})
        assert not result.ok
        assert "nothing is staged" in result.output.lower()

    def test_valid_commit_proposed(self, project, repo):
        (repo / "README.md").write_text("v2\n")
        run_git(repo, "add", "README.md")
        result = build_git_commit_tool(project).execute({"message": "Update README"})
        assert result.ok
        op = result.pending_git_operation
        assert op.kind == "commit"
        assert op.message == "Update README"
        assert op.expected_staged_files == ["README.md"]

    def test_empty_message_rejected(self, project, repo):
        (repo / "README.md").write_text("v2\n")
        run_git(repo, "add", "README.md")
        result = build_git_commit_tool(project).execute({"message": "   "})
        assert not result.ok

    def test_non_repo_reported_gracefully(self, non_git_project):
        result = build_git_commit_tool(non_git_project).execute({"message": "x"})
        assert not result.ok
        assert result.output == "This project is not a Git repository."


class TestApplyGitOperation:
    def test_create_branch_succeeds(self, repo):
        from agent.git_policy import ProposedGitOperation

        op = ProposedGitOperation(kind="create_branch", repo_root=repo, branch_name="feature/new")
        result = apply_git_operation(op)
        assert result.ok
        branches = subprocess.run(
            ["git", "branch", "--list", "feature/new"], cwd=str(repo), capture_output=True, text=True
        ).stdout
        assert "feature/new" in branches

    def test_create_branch_does_not_check_it_out(self, repo):
        from agent.git_policy import ProposedGitOperation

        before = subprocess.run(
            ["git", "branch", "--show-current"], cwd=str(repo), capture_output=True, text=True
        ).stdout.strip()
        op = ProposedGitOperation(kind="create_branch", repo_root=repo, branch_name="feature/new")
        apply_git_operation(op)
        after = subprocess.run(
            ["git", "branch", "--show-current"], cwd=str(repo), capture_output=True, text=True
        ).stdout.strip()
        assert before == after

    def test_stage_succeeds(self, project, repo):
        (repo / "README.md").write_text("v2\n")
        propose = build_git_stage_tool(project).execute({"paths": ["README.md"]})
        result = apply_git_operation(propose.pending_git_operation)
        assert result.ok
        staged = subprocess.run(
            ["git", "diff", "--staged", "--name-only"], cwd=str(repo), capture_output=True, text=True
        ).stdout
        assert "README.md" in staged

    def test_commit_succeeds(self, project, repo):
        (repo / "README.md").write_text("v2\n")
        run_git(repo, "add", "README.md")
        propose = build_git_commit_tool(project).execute({"message": "Update README"})
        result = apply_git_operation(propose.pending_git_operation)
        assert result.ok
        assert "Update README" in result.output
        log = subprocess.run(
            ["git", "log", "-1", "--pretty=%s"], cwd=str(repo), capture_output=True, text=True
        ).stdout.strip()
        assert log == "Update README"

    def test_commit_refuses_if_staged_set_changed_since_proposal(self, project, repo):
        (repo / "README.md").write_text("v2\n")
        run_git(repo, "add", "README.md")
        propose = build_git_commit_tool(project).execute({"message": "Update README"})

        # Staging area changes after the proposal but before approval/apply.
        (repo / "backend" / "auth.py").write_text("def login():\n    return None\n")
        run_git(repo, "add", "backend/auth.py")

        result = apply_git_operation(propose.pending_git_operation)
        assert not result.ok
        assert "changed since this commit was proposed" in result.output

        # Nothing should have been committed.
        log = subprocess.run(
            ["git", "log", "-1", "--pretty=%s"], cwd=str(repo), capture_output=True, text=True
        ).stdout.strip()
        assert log == "Initial commit"

    def test_commit_only_stages_requested_file_leaves_others_untouched(self, project, repo):
        """Mandatory Phase 5 test: staging one file must never sweep in
        unrelated modified files."""
        (repo / "README.md").write_text("v2\n")
        (repo / "backend" / "auth.py").write_text("def login():\n    return None\n")

        propose = build_git_stage_tool(project).execute({"paths": ["README.md"]})
        apply_git_operation(propose.pending_git_operation)

        staged = subprocess.run(
            ["git", "diff", "--staged", "--name-only"], cwd=str(repo), capture_output=True, text=True
        ).stdout.split()
        unstaged = subprocess.run(
            ["git", "diff", "--name-only"], cwd=str(repo), capture_output=True, text=True
        ).stdout.split()

        assert staged == ["README.md"]
        assert unstaged == ["backend/auth.py"]

    def test_apply_when_repo_vanishes_is_refused(self, repo):
        from agent.git_policy import ProposedGitOperation
        import shutil

        op = ProposedGitOperation(kind="create_branch", repo_root=repo, branch_name="feature/x")
        shutil.rmtree(repo / ".git")
        result = apply_git_operation(op)
        assert not result.ok
        assert "not a git repository" in result.output.lower()


class TestNoGenericGitCommandTool:
    """The most important Phase 5 security property: no tool exists that
    accepts an arbitrary Git command or subcommand string."""

    def test_registry_only_exposes_the_six_narrow_git_tools(self):
        import inspect

        import agent.tools.git as git_module

        tool_builders = [
            name for name, obj in inspect.getmembers(git_module)
            if name.startswith("build_git_") and inspect.isfunction(obj)
        ]
        assert set(tool_builders) == {
            "build_git_status_tool",
            "build_git_diff_tool",
            "build_git_log_tool",
            "build_git_create_branch_tool",
            "build_git_stage_tool",
            "build_git_commit_tool",
        }

    def test_no_run_git_command_function_exists(self):
        import agent.tools.git as git_module

        assert not hasattr(git_module, "run_git_command")
        assert not hasattr(git_module, "run_command")
