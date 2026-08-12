"""Tests for the Git validation layer in agent/git_policy.py.

Branch-name validation calls the real `git check-ref-format` binary (it's a
fast, pure syntax checker that doesn't need a repository or touch any
state), but nothing here ever creates a branch, stages a file, or commits.
"""
from __future__ import annotations

import pytest

from agent.git_policy import (
    MAX_COMMIT_MESSAGE_CHARS,
    GitPolicyError,
    validate_branch_name,
    validate_commit_message,
    validate_stage_paths,
)
from agent.project import ProjectRoot


@pytest.fixture
def project(tmp_path):
    (tmp_path / "backend").mkdir()
    (tmp_path / "backend" / "auth.py").write_text("x = 1\n")
    (tmp_path / ".env").write_text("SECRET=1\n")
    return ProjectRoot(tmp_path)


class TestValidateBranchName:
    def test_simple_name_accepted(self, tmp_path):
        assert validate_branch_name(tmp_path, "feature-x") == "feature-x"

    def test_name_with_slash_accepted(self, tmp_path):
        assert validate_branch_name(tmp_path, "fix/login-authentication") == "fix/login-authentication"

    def test_strips_whitespace(self, tmp_path):
        assert validate_branch_name(tmp_path, "  feature-x  ") == "feature-x"

    def test_empty_rejected(self, tmp_path):
        with pytest.raises(GitPolicyError):
            validate_branch_name(tmp_path, "")

    def test_whitespace_only_rejected(self, tmp_path):
        with pytest.raises(GitPolicyError):
            validate_branch_name(tmp_path, "   ")

    def test_leading_dash_rejected(self, tmp_path):
        with pytest.raises(GitPolicyError):
            validate_branch_name(tmp_path, "-force")

    def test_shell_metacharacters_rejected(self, tmp_path):
        with pytest.raises(GitPolicyError):
            validate_branch_name(tmp_path, "feature; rm -rf .")

    def test_command_substitution_rejected(self, tmp_path):
        with pytest.raises(GitPolicyError):
            validate_branch_name(tmp_path, "feature-$(whoami)")

    def test_double_dot_rejected(self, tmp_path):
        with pytest.raises(GitPolicyError):
            validate_branch_name(tmp_path, "feature..x")

    def test_trailing_dot_rejected(self, tmp_path):
        with pytest.raises(GitPolicyError):
            validate_branch_name(tmp_path, "feature.")

    def test_trailing_slash_rejected(self, tmp_path):
        with pytest.raises(GitPolicyError):
            validate_branch_name(tmp_path, "feature/")

    def test_lock_suffix_rejected(self, tmp_path):
        with pytest.raises(GitPolicyError):
            validate_branch_name(tmp_path, "feature.lock")

    def test_space_rejected(self, tmp_path):
        with pytest.raises(GitPolicyError):
            validate_branch_name(tmp_path, "feature name")

    def test_leading_dot_rejected_by_git_check_ref_format(self, tmp_path):
        with pytest.raises(GitPolicyError):
            validate_branch_name(tmp_path, ".feature")


class TestValidateStagePaths:
    def test_valid_relative_paths_accepted(self, project):
        result = validate_stage_paths(project, ["backend/auth.py"])
        assert result == ["backend/auth.py"]

    def test_multiple_paths_accepted(self, project):
        (project.root / "README.md").write_text("x")
        result = validate_stage_paths(project, ["backend/auth.py", "README.md"])
        assert set(result) == {"backend/auth.py", "README.md"}

    def test_empty_list_rejected(self, project):
        with pytest.raises(GitPolicyError):
            validate_stage_paths(project, [])

    def test_path_traversal_rejected(self, project):
        with pytest.raises(GitPolicyError):
            validate_stage_paths(project, ["../../etc/passwd"])

    def test_absolute_path_outside_project_rejected(self, project):
        with pytest.raises(GitPolicyError):
            validate_stage_paths(project, ["/etc/passwd"])

    def test_leading_dash_rejected(self, project):
        with pytest.raises(GitPolicyError):
            validate_stage_paths(project, ["--force"])

    def test_shell_metacharacters_rejected(self, project):
        with pytest.raises(GitPolicyError):
            validate_stage_paths(project, ["backend/auth.py; rm -rf ."])

    def test_empty_string_path_rejected(self, project):
        with pytest.raises(GitPolicyError):
            validate_stage_paths(project, [""])

    def test_one_bad_path_rejects_whole_batch(self, project):
        with pytest.raises(GitPolicyError):
            validate_stage_paths(project, ["backend/auth.py", "../../etc/passwd"])


class TestValidateCommitMessage:
    def test_normal_message_accepted(self):
        assert validate_commit_message("Fix JWT authentication flow") == "Fix JWT authentication flow"

    def test_message_with_ampersand_accepted(self):
        """Commit messages are opaque text passed as a single argv value --
        they must not be over-restricted the way shell-parsed arguments are."""
        assert validate_commit_message("Fix user & admin role conflict") == "Fix user & admin role conflict"

    def test_message_with_semicolon_accepted(self):
        assert validate_commit_message("Fix bug; update docs") == "Fix bug; update docs"

    def test_message_with_quotes_accepted(self):
        assert validate_commit_message('Fix "login" bug') == 'Fix "login" bug'

    def test_strips_surrounding_whitespace(self):
        assert validate_commit_message("  Fix bug  ") == "Fix bug"

    def test_empty_message_rejected(self):
        with pytest.raises(GitPolicyError):
            validate_commit_message("")

    def test_whitespace_only_message_rejected(self):
        with pytest.raises(GitPolicyError):
            validate_commit_message("   ")

    def test_absurdly_long_message_rejected(self):
        with pytest.raises(GitPolicyError):
            validate_commit_message("x" * (MAX_COMMIT_MESSAGE_CHARS + 1))

    def test_message_at_max_length_accepted(self):
        message = "x" * MAX_COMMIT_MESSAGE_CHARS
        assert validate_commit_message(message) == message

    def test_nul_byte_rejected(self):
        with pytest.raises(GitPolicyError):
            validate_commit_message("bad\x00message")
