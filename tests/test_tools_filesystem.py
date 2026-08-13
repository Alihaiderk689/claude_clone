"""Tests for the list_files and read_file tools."""
from __future__ import annotations

import os

import pytest

from agent.project import MAX_FILE_SIZE_BYTES, ProjectRoot
from agent.tools.filesystem import (
    ListFilesArgs,
    ReadFileArgs,
    build_list_files_tool,
    build_read_file_tool,
)

requires_non_root = pytest.mark.skipif(
    hasattr(os, "getuid") and os.getuid() == 0,
    reason="permission checks are bypassed when running as root",
)


@pytest.fixture
def sample_project(tmp_path):
    (tmp_path / "backend" / "accounts").mkdir(parents=True)
    (tmp_path / "backend" / "accounts" / "views.py").write_text(
        "\n".join(f"line {i}" for i in range(1, 11)) + "\n"
    )
    (tmp_path / "backend" / "manage.py").write_text("# manage\n")
    (tmp_path / "README.md").write_text("# Sample project\n")

    (tmp_path / "node_modules" / "pkg").mkdir(parents=True)
    (tmp_path / "node_modules" / "pkg" / "index.js").write_text("module.exports = {};\n")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("[core]\n")

    (tmp_path / ".env").write_text("SECRET=supersecret\n")
    (tmp_path / "id_rsa").write_text("-----BEGIN OPENSSH PRIVATE KEY-----\nshh\n")

    (tmp_path / "big.txt").write_text("x" * (MAX_FILE_SIZE_BYTES + 1))
    (tmp_path / "binary.dat").write_bytes(bytes(range(256)) * 4)

    return tmp_path


@pytest.fixture
def project(sample_project):
    return ProjectRoot(sample_project)


@pytest.fixture
def list_files_tool(project):
    return build_list_files_tool(project)


@pytest.fixture
def read_file_tool(project):
    return build_read_file_tool(project)


class TestListFiles:
    def test_lists_project_files(self, list_files_tool):
        result = list_files_tool.execute({"path": "."})
        assert result.ok
        assert "README.md" in result.output
        assert "backend/" in result.output
        assert "backend/manage.py" in result.output
        assert "backend/accounts/" in result.output

    def test_ignores_common_directories(self, list_files_tool):
        result = list_files_tool.execute({"path": "."})
        assert "node_modules" not in result.output
        assert ".git" not in result.output

    def test_hides_sensitive_files_from_listing(self, list_files_tool):
        result = list_files_tool.execute({"path": "."})
        assert "id_rsa" not in result.output

    def test_env_file_is_not_hidden(self, list_files_tool):
        """.env is deliberately not treated as sensitive (by request) --
        see agent/project.py's ENV_FILE_PATTERNS."""
        result = list_files_tool.execute({"path": "."})
        assert ".env" in result.output

    def test_lists_subdirectory(self, list_files_tool):
        result = list_files_tool.execute({"path": "backend/accounts"})
        assert result.ok
        assert "views.py" in result.output

    def test_nonexistent_path(self, list_files_tool):
        result = list_files_tool.execute({"path": "does/not/exist"})
        assert not result.ok
        assert "not found" in result.output.lower()

    def test_path_is_a_file_not_a_directory(self, list_files_tool):
        result = list_files_tool.execute({"path": "README.md"})
        assert not result.ok
        assert "not a directory" in result.output.lower()

    def test_rejects_path_traversal(self, list_files_tool):
        result = list_files_tool.execute({"path": "../"})
        assert not result.ok
        assert "access denied" in result.output.lower()

    def test_respects_max_entries(self, project, tmp_path):
        many_dir = project.root / "many"
        many_dir.mkdir()
        for i in range(20):
            (many_dir / f"file_{i}.txt").write_text("x")
        tool = build_list_files_tool(project)
        result = tool.execute({"path": "many", "max_entries": 5})
        assert result.ok
        listed = [line for line in result.output.splitlines() if not line.startswith("...")]
        assert len(listed) == 5
        assert "truncated" in result.output.lower()


class TestReadFile:
    def test_reads_whole_small_file(self, read_file_tool):
        result = read_file_tool.execute({"path": "README.md"})
        assert result.ok
        assert "Sample project" in result.output

    def test_reads_line_range(self, read_file_tool):
        result = read_file_tool.execute(
            {"path": "backend/accounts/views.py", "start_line": 2, "end_line": 4}
        )
        assert result.ok
        assert "line 2" in result.output
        assert "line 4" in result.output
        assert "line 5" not in result.output
        assert "line 1" not in result.output

    def test_nonexistent_file(self, read_file_tool):
        result = read_file_tool.execute({"path": "nope.txt"})
        assert not result.ok
        assert "not found" in result.output.lower()

    def test_rejects_directory(self, read_file_tool):
        result = read_file_tool.execute({"path": "backend"})
        assert not result.ok
        assert "directory" in result.output.lower()

    def test_rejects_oversized_file_without_range(self, read_file_tool):
        result = read_file_tool.execute({"path": "big.txt"})
        assert not result.ok
        assert "exceeds" in result.output.lower() or "limit" in result.output.lower()

    def test_allows_oversized_file_with_range(self, read_file_tool):
        result = read_file_tool.execute({"path": "big.txt", "start_line": 1, "end_line": 1})
        assert result.ok

    def test_rejects_sensitive_file(self, read_file_tool):
        result = read_file_tool.execute({"path": "id_rsa"})
        assert not result.ok
        assert "sensitive" in result.output.lower()

    def test_env_file_is_readable(self, read_file_tool):
        result = read_file_tool.execute({"path": ".env"})
        assert result.ok

    def test_rejects_path_traversal(self, read_file_tool):
        result = read_file_tool.execute({"path": "../../etc/passwd"})
        assert not result.ok
        assert "access denied" in result.output.lower()

    def test_handles_binary_file_gracefully(self, read_file_tool):
        result = read_file_tool.execute({"path": "binary.dat"})
        assert not result.ok
        assert "binary" in result.output.lower()

    def test_invalid_arguments_are_rejected(self, read_file_tool):
        result = read_file_tool.execute({})
        assert not result.ok

    def test_not_found_is_classified(self, read_file_tool):
        result = read_file_tool.execute({"path": "does_not_exist.py"})
        assert not result.ok
        assert result.error_type == "NotFoundError"
        assert result.recoverable is True

    @requires_non_root
    def test_permission_denied_is_classified_not_crashed(self, sample_project, read_file_tool):
        target = sample_project / "README.md"
        target.chmod(0o000)
        try:
            result = read_file_tool.execute({"path": "README.md"})
        finally:
            target.chmod(0o644)  # restore so tmp_path cleanup can remove it

        assert not result.ok  # must not raise
        assert result.error_type == "PermissionError"
        assert result.recoverable is False
        assert "permission" in result.output.lower()


class TestListFilesPermissionError:
    @requires_non_root
    def test_permission_denied_directory_is_classified_not_crashed(self, tmp_path):
        (tmp_path / "locked").mkdir()
        (tmp_path / "locked" / "secret.txt").write_text("x\n")
        (tmp_path / "locked").chmod(0o000)
        project = ProjectRoot(tmp_path)
        tool = build_list_files_tool(project)

        try:
            result = tool.execute({"path": "locked"})
        finally:
            (tmp_path / "locked").chmod(0o755)

        assert not result.ok  # must not raise
        assert result.error_type == "PermissionError"
