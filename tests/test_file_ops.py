"""Tests for the delete/rename validation layer in agent/file_ops.py.

Pure validation -- nothing here ever touches the filesystem (no delete, no
rename). See tests/test_tools_file_ops.py for the tool + apply_file_op layer.
"""
from __future__ import annotations

import pytest

from agent.file_ops import FileOpError, validate_delete, validate_rename
from agent.project import ProjectRoot


@pytest.fixture
def project(tmp_path):
    (tmp_path / "backend").mkdir()
    (tmp_path / "backend" / "old.py").write_text("x = 1\n")
    (tmp_path / "backend" / "existing.py").write_text("y = 2\n")
    (tmp_path / "id_rsa").write_text("-----BEGIN OPENSSH PRIVATE KEY-----\nshh\n")
    return ProjectRoot(tmp_path)


class TestValidateDelete:
    def test_valid_delete_is_accepted(self, project):
        op = validate_delete(project, "backend/old.py")
        assert op.kind == "delete"
        assert op.source_path == "backend/old.py"
        assert op.resolved_source == project.root / "backend" / "old.py"

    def test_nonexistent_file_rejected(self, project):
        with pytest.raises(FileOpError, match="not found"):
            validate_delete(project, "backend/nope.py")

    def test_directory_rejected(self, project):
        with pytest.raises(FileOpError, match="directory"):
            validate_delete(project, "backend")

    def test_project_root_rejected(self, project):
        with pytest.raises(FileOpError):
            validate_delete(project, ".")

    def test_empty_path_rejected(self, project):
        with pytest.raises(FileOpError):
            validate_delete(project, "")

    def test_path_traversal_rejected(self, project):
        with pytest.raises(FileOpError, match="outside the project root"):
            validate_delete(project, "../../etc/passwd")

    def test_absolute_path_outside_project_rejected(self, project):
        with pytest.raises(FileOpError, match="outside the project root"):
            validate_delete(project, "/etc/passwd")

    def test_sensitive_file_rejected(self, project):
        with pytest.raises(FileOpError, match="sensitive"):
            validate_delete(project, "id_rsa")


class TestValidateRename:
    def test_valid_rename_is_accepted(self, project):
        op = validate_rename(project, "backend/old.py", "backend/new.py")
        assert op.kind == "rename"
        assert op.source_path == "backend/old.py"
        assert op.destination_path == "backend/new.py"
        assert op.resolved_destination == project.root / "backend" / "new.py"

    def test_rename_into_new_directory_is_accepted(self, project):
        op = validate_rename(project, "backend/old.py", "lib/moved.py")
        assert op.resolved_destination == project.root / "lib" / "moved.py"

    def test_nonexistent_source_rejected(self, project):
        with pytest.raises(FileOpError, match="not found"):
            validate_rename(project, "backend/nope.py", "backend/new.py")

    def test_existing_destination_rejected(self, project):
        with pytest.raises(FileOpError, match="already exists"):
            validate_rename(project, "backend/old.py", "backend/existing.py")

    def test_source_is_directory_rejected(self, project):
        with pytest.raises(FileOpError, match="directory"):
            validate_rename(project, "backend", "elsewhere")

    def test_same_source_and_destination_rejected(self, project):
        with pytest.raises(FileOpError, match="same path"):
            validate_rename(project, "backend/old.py", "backend/old.py")

    def test_source_traversal_rejected(self, project):
        with pytest.raises(FileOpError, match="outside the project root"):
            validate_rename(project, "../../etc/passwd", "backend/new.py")

    def test_destination_traversal_rejected(self, project):
        with pytest.raises(FileOpError, match="outside the project root"):
            validate_rename(project, "backend/old.py", "../../evil.py")

    def test_project_root_as_source_rejected(self, project):
        with pytest.raises(FileOpError):
            validate_rename(project, ".", "backend/new.py")

    def test_sensitive_source_rejected(self, project):
        with pytest.raises(FileOpError, match="sensitive"):
            validate_rename(project, "id_rsa", "backend/moved_key")

    def test_sensitive_destination_rejected(self, project):
        with pytest.raises(FileOpError, match="sensitive"):
            validate_rename(project, "backend/old.py", "backend/id_rsa")

    def test_missing_source_or_destination_rejected(self, project):
        with pytest.raises(FileOpError):
            validate_rename(project, "", "backend/new.py")
        with pytest.raises(FileOpError):
            validate_rename(project, "backend/old.py", "")
