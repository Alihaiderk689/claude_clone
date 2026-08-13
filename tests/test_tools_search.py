"""Tests for the search_files tool: fallback scanner and (mocked) ripgrep path."""
from __future__ import annotations

from unittest import mock

import pytest

from agent.project import ProjectRoot
from agent.tools.search import build_search_files_tool


@pytest.fixture
def sample_project(tmp_path):
    (tmp_path / "backend" / "accounts").mkdir(parents=True)
    (tmp_path / "backend" / "accounts" / "views.py").write_text(
        "class GoogleLoginView(APIView):\n    def post(self, request):\n        pass\n"
    )
    (tmp_path / "backend" / "accounts" / "serializers.py").write_text(
        "class GoogleLoginSerializer(serializers.Serializer):\n    pass\n"
    )
    (tmp_path / "README.md").write_text("This project uses JWT authentication.\n")

    (tmp_path / "node_modules" / "pkg").mkdir(parents=True)
    (tmp_path / "node_modules" / "pkg" / "index.js").write_text("// JWT mentioned here too\n")

    (tmp_path / ".env").write_text("JWT_SECRET=whatever\n")
    (tmp_path / "id_rsa").write_text("-----BEGIN OPENSSH PRIVATE KEY-----\nJWT_SECRET too\n")

    return tmp_path


@pytest.fixture
def project(sample_project):
    return ProjectRoot(sample_project)


@pytest.fixture
def search_tool(project):
    return build_search_files_tool(project)


class TestSearchFallback:
    """These exercise the pure-Python fallback (ripgrep unavailable in CI)."""

    def test_finds_matching_text(self, search_tool):
        result = search_tool.execute({"query": "GoogleLoginView"})
        assert result.ok
        assert "backend/accounts/views.py" in result.output
        assert "GoogleLoginView" in result.output

    def test_no_matches(self, search_tool):
        result = search_tool.execute({"query": "ThisStringDoesNotExistAnywhere"})
        assert result.ok
        assert "no matches" in result.output.lower()

    def test_ignores_node_modules(self, search_tool):
        result = search_tool.execute({"query": "JWT"})
        assert result.ok
        assert "node_modules" not in result.output

    def test_skips_sensitive_files(self, search_tool):
        result = search_tool.execute({"query": "JWT_SECRET"})
        assert result.ok
        assert "id_rsa" not in result.output

    def test_does_not_skip_env_files(self, search_tool):
        result = search_tool.execute({"query": "JWT_SECRET"})
        assert result.ok
        assert ".env" in result.output

    def test_respects_max_results(self, project):
        many_dir = project.root / "many"
        many_dir.mkdir()
        for i in range(10):
            (many_dir / f"f{i}.py").write_text("TARGET\n")
        tool = build_search_files_tool(project)
        result = tool.execute({"query": "TARGET", "path": "many", "max_results": 3})
        assert result.ok
        match_lines = [l for l in result.output.splitlines() if l.startswith("many/")]
        assert len(match_lines) == 3

    def test_rejects_path_traversal(self, search_tool):
        result = search_tool.execute({"query": "x", "path": "../"})
        assert not result.ok
        assert "access denied" in result.output.lower()

    def test_rejects_empty_query(self, search_tool):
        result = search_tool.execute({"query": "   "})
        assert not result.ok

    def test_nonexistent_path(self, search_tool):
        result = search_tool.execute({"query": "x", "path": "does/not/exist"})
        assert not result.ok
        assert "not found" in result.output.lower()


class TestSearchRipgrepPath:
    """Mock shutil.which/subprocess.run to exercise the ripgrep branch without
    depending on ripgrep actually being installed on the test machine."""

    def test_uses_ripgrep_when_available(self, project):
        fake_rg_output = "backend/accounts/views.py:1:class GoogleLoginView(APIView):\n"

        with mock.patch("agent.tools.search.shutil.which", return_value="/usr/bin/rg"), \
             mock.patch("agent.tools.search.subprocess.run") as mock_run:
            mock_run.return_value = mock.Mock(
                returncode=0,
                stdout=str(project.root / fake_rg_output),
                stderr="",
            )
            tool = build_search_files_tool(project)
            result = tool.execute({"query": "GoogleLoginView"})

        assert result.ok
        assert "GoogleLoginView" in result.output
        mock_run.assert_called_once()

    def test_ripgrep_no_matches_returncode_1(self, project):
        with mock.patch("agent.tools.search.shutil.which", return_value="/usr/bin/rg"), \
             mock.patch("agent.tools.search.subprocess.run") as mock_run:
            mock_run.return_value = mock.Mock(returncode=1, stdout="", stderr="")
            tool = build_search_files_tool(project)
            result = tool.execute({"query": "NoSuchThing"})

        assert result.ok
        assert "no matches" in result.output.lower()

    def test_ripgrep_real_error_surfaces(self, project):
        with mock.patch("agent.tools.search.shutil.which", return_value="/usr/bin/rg"), \
             mock.patch("agent.tools.search.subprocess.run") as mock_run:
            mock_run.return_value = mock.Mock(returncode=2, stdout="", stderr="bad pattern")
            tool = build_search_files_tool(project)
            result = tool.execute({"query": "("})

        assert not result.ok
        assert "ripgrep error" in result.output.lower()
