"""Tests for delete_file/rename_file and apply_file_op.

Same propose/apply split as tests/test_tools_editing.py verifies for
edit_file/write_file: a tool's run() must only validate and build a
ProposedFileOp in memory, never touch disk, and apply_file_op() must only be
reachable after that -- never on its own.
"""
from __future__ import annotations

from unittest import mock

import pytest

from agent.project import ProjectRoot
from agent.tools.file_ops import apply_file_op, build_delete_file_tool, build_rename_file_tool
from agent.tools.state import FileStateTracker


@pytest.fixture
def sample_project(tmp_path):
    (tmp_path / "backend").mkdir()
    (tmp_path / "backend" / "old.py").write_text("x = 1\n")
    (tmp_path / "backend" / "existing.py").write_text("y = 2\n")
    return tmp_path


@pytest.fixture
def project(sample_project):
    return ProjectRoot(sample_project)


@pytest.fixture
def tracker():
    return FileStateTracker()


@pytest.fixture
def delete_tool(project):
    return build_delete_file_tool(project)


@pytest.fixture
def rename_tool(project):
    return build_rename_file_tool(project)


class TestDeleteFilePropose:
    def test_valid_delete_proposes_without_deleting(self, delete_tool, project):
        result = delete_tool.execute({"path": "backend/old.py"})
        assert result.ok
        assert result.pending_file_op is not None
        assert result.pending_file_op.kind == "delete"
        assert (project.root / "backend" / "old.py").exists()  # untouched

    def test_nonexistent_file(self, delete_tool):
        result = delete_tool.execute({"path": "backend/nope.py"})
        assert not result.ok
        assert result.error_type == "NotFoundError"

    def test_directory_rejected(self, delete_tool):
        result = delete_tool.execute({"path": "backend"})
        assert not result.ok

    def test_path_traversal_rejected(self, delete_tool):
        result = delete_tool.execute({"path": "../../etc/passwd"})
        assert not result.ok

    def test_malformed_arguments_rejected(self, delete_tool):
        result = delete_tool.execute({})
        assert not result.ok


class TestRenameFilePropose:
    def test_valid_rename_proposes_without_renaming(self, rename_tool, project):
        result = rename_tool.execute({"source": "backend/old.py", "destination": "backend/new.py"})
        assert result.ok
        assert result.pending_file_op is not None
        assert result.pending_file_op.kind == "rename"
        assert (project.root / "backend" / "old.py").exists()  # untouched
        assert not (project.root / "backend" / "new.py").exists()

    def test_existing_destination_rejected(self, rename_tool):
        result = rename_tool.execute(
            {"source": "backend/old.py", "destination": "backend/existing.py"}
        )
        assert not result.ok
        assert "already exists" in result.output.lower()

    def test_malformed_arguments_rejected(self, rename_tool):
        result = rename_tool.execute({"source": "backend/old.py"})  # missing destination
        assert not result.ok


class TestApplyFileOp:
    def test_approved_delete_removes_file(self, delete_tool, project, tracker):
        propose = delete_tool.execute({"path": "backend/old.py"})
        assert propose.ok

        apply_result = apply_file_op(propose.pending_file_op, tracker)

        assert apply_result.ok
        assert "deleted" in apply_result.output.lower()
        assert not (project.root / "backend" / "old.py").exists()

    def test_delete_forgets_tracker_entry(self, project, tracker):
        target = project.root / "backend" / "old.py"
        tracker.record(target, target.read_text())
        delete_tool = build_delete_file_tool(project)
        propose = delete_tool.execute({"path": "backend/old.py"})

        apply_file_op(propose.pending_file_op, tracker)

        assert tracker.is_fresh(target, "anything at all")  # nothing recorded anymore

    def test_approved_rename_moves_file(self, rename_tool, project, tracker):
        propose = rename_tool.execute({"source": "backend/old.py", "destination": "backend/new.py"})
        assert propose.ok

        apply_result = apply_file_op(propose.pending_file_op, tracker)

        assert apply_result.ok
        assert "renamed" in apply_result.output.lower()
        assert not (project.root / "backend" / "old.py").exists()
        assert (project.root / "backend" / "new.py").read_text() == "x = 1\n"

    def test_rename_creates_missing_parent_directories(self, rename_tool, project, tracker):
        propose = rename_tool.execute({"source": "backend/old.py", "destination": "lib/moved/old.py"})
        apply_result = apply_file_op(propose.pending_file_op, tracker)
        assert apply_result.ok
        assert (project.root / "lib" / "moved" / "old.py").read_text() == "x = 1\n"

    def test_rename_forgets_tracker_entry_for_source(self, project, tracker):
        target = project.root / "backend" / "old.py"
        tracker.record(target, target.read_text())
        rename_tool = build_rename_file_tool(project)
        propose = rename_tool.execute({"source": "backend/old.py", "destination": "backend/new.py"})

        apply_file_op(propose.pending_file_op, tracker)

        assert tracker.is_fresh(target, "anything at all")

    def test_delete_of_already_missing_file_reports_failure_not_crash(self, delete_tool, project, tracker):
        propose = delete_tool.execute({"path": "backend/old.py"})
        (project.root / "backend" / "old.py").unlink()  # removed out from under it

        apply_result = apply_file_op(propose.pending_file_op, tracker)

        assert not apply_result.ok
        assert "no longer exists" in apply_result.output.lower()

    def test_rename_race_destination_now_exists_reports_failure_not_overwrite(
        self, rename_tool, project, tracker
    ):
        propose = rename_tool.execute({"source": "backend/old.py", "destination": "backend/new.py"})
        (project.root / "backend" / "new.py").write_text("raced in\n")  # appeared after proposal

        apply_result = apply_file_op(propose.pending_file_op, tracker)

        assert not apply_result.ok
        assert (project.root / "backend" / "new.py").read_text() == "raced in\n"  # not overwritten
        assert (project.root / "backend" / "old.py").exists()  # source untouched

    def test_delete_permission_error_reports_failure_not_crash(self, delete_tool, project, tracker):
        propose = delete_tool.execute({"path": "backend/old.py"})
        with mock.patch("agent.tools.file_ops.os.remove", side_effect=PermissionError("denied")):
            apply_result = apply_file_op(propose.pending_file_op, tracker)
        assert not apply_result.ok
        assert apply_result.error_type == "PermissionError"
        assert (project.root / "backend" / "old.py").exists()
