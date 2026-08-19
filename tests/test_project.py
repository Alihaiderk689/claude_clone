"""Tests for ProjectRoot path resolution and traversal protection."""
from __future__ import annotations

import pytest

from agent.project import PathSecurityError, ProjectRoot


@pytest.fixture
def project(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("print('hi')\n")
    outside = tmp_path.parent / "outside_secret.txt"
    outside.write_text("top secret\n")
    return ProjectRoot(tmp_path)


def test_resolves_normal_relative_path(project):
    resolved = project.resolve("src/app.py")
    assert resolved == project.root / "src" / "app.py"


def test_resolves_dot_to_root(project):
    assert project.resolve(".") == project.root
    assert project.resolve("") == project.root
    assert project.resolve(None) == project.root


def test_rejects_parent_traversal(project):
    with pytest.raises(PathSecurityError):
        project.resolve("../../private_file")


def test_rejects_traversal_hidden_inside_a_deeper_path(project):
    with pytest.raises(PathSecurityError):
        project.resolve("src/../../outside_secret.txt")


def test_rejects_absolute_path_outside_root(project):
    outside_abs = str(project.root.parent / "outside_secret.txt")
    with pytest.raises(PathSecurityError):
        project.resolve(outside_abs)


def test_allows_absolute_path_inside_root(project):
    inside_abs = str(project.root / "src" / "app.py")
    resolved = project.resolve(inside_abs)
    assert resolved == project.root / "src" / "app.py"


def test_rejects_unfilled_placeholder_path(project):
    with pytest.raises(PathSecurityError):
        project.resolve("<file_path>")


def test_rejects_placeholder_embedded_in_a_longer_path(project):
    with pytest.raises(PathSecurityError):
        project.resolve("src/<something>")


def test_is_ignored_dir(project):
    assert project.is_ignored_dir(".git")
    assert project.is_ignored_dir("node_modules")
    assert not project.is_ignored_dir("src")


def test_is_sensitive_no_longer_matches_env_files(project):
    """.env is deliberately NOT in is_sensitive() -- the agent is allowed
    to create/edit .env directly (by request). It's still flagged
    separately by looks_like_env_file(), used only for git.py's
    stage-time warning -- see test_looks_like_env_file below."""
    assert not project.is_sensitive(project.root / ".env")
    assert not project.is_sensitive(project.root / ".env.local")
    assert project.is_sensitive(project.root / "id_rsa")
    assert not project.is_sensitive(project.root / "src" / "app.py")


def test_looks_like_env_file(project):
    assert project.looks_like_env_file(project.root / ".env")
    assert project.looks_like_env_file(project.root / ".env.local")
    assert not project.looks_like_env_file(project.root / "id_rsa")
    assert not project.looks_like_env_file(project.root / "src" / "app.py")
