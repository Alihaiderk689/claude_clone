"""Tests for edit_file, write_file, and apply_change.

These verify the propose/apply split at the heart of Phase 3: run() must
only validate and build a ProposedChange in memory, never touch disk, and
apply_change() must only be reachable after that -- never on its own.
"""
from __future__ import annotations

import os
from pathlib import Path
from unittest import mock

import pytest

from agent.project import HARD_MAX_FILE_SIZE_BYTES, ProjectRoot
from agent.tools.editing import (
    apply_change,
    build_edit_file_tool,
    build_write_file_tool,
)
from agent.tools.state import FileStateTracker


@pytest.fixture
def sample_project(tmp_path):
    (tmp_path / "backend").mkdir()
    (tmp_path / "backend" / "auth.py").write_text(
        "def login(username, password):\n"
        "    if username == 'admin' and password == '1234':\n"
        "        return {'token': 'abc123'}\n"
        "    return None\n"
    )
    (tmp_path / "backend" / "dup.py").write_text("x = 1\nx = 1\n")
    (tmp_path / ".env").write_text("SECRET=shh\n")
    (tmp_path / "id_rsa").write_text("-----BEGIN OPENSSH PRIVATE KEY-----\nshh\n")
    return tmp_path


@pytest.fixture
def project(sample_project):
    return ProjectRoot(sample_project)


@pytest.fixture
def tracker():
    return FileStateTracker()


@pytest.fixture
def edit_tool(project, tracker):
    return build_edit_file_tool(project, tracker)


@pytest.fixture
def write_tool(project, tracker):
    return build_write_file_tool(project, tracker)


class TestEditFilePropose:
    def test_valid_edit_proposes_without_writing(self, edit_tool, project):
        original = (project.root / "backend" / "auth.py").read_text()

        result = edit_tool.execute(
            {"path": "backend/auth.py", "old_text": "return None", "new_text": "return {'error': 'bad'}"}
        )

        assert result.ok
        assert result.pending_change is not None
        change = result.pending_change
        assert change.kind == "edit"
        assert "return {'error': 'bad'}" in change.new_content
        assert "return None" in change.old_content
        # Nothing written yet.
        assert (project.root / "backend" / "auth.py").read_text() == original

    def test_nonexistent_file(self, edit_tool):
        result = edit_tool.execute({"path": "nope.py", "old_text": "x", "new_text": "y"})
        assert not result.ok
        assert "not found" in result.output.lower()
        assert result.pending_change is None

    def test_target_text_not_found(self, edit_tool):
        result = edit_tool.execute(
            {"path": "backend/auth.py", "old_text": "totally not in the file", "new_text": "y"}
        )
        assert not result.ok
        assert "not found" in result.output.lower()

    def test_target_text_appears_multiple_times_is_rejected(self, edit_tool):
        result = edit_tool.execute({"path": "backend/dup.py", "old_text": "x = 1", "new_text": "x = 2"})
        assert not result.ok
        assert "ambiguous" in result.output.lower() or "2 times" in result.output

    def test_path_traversal_rejected(self, edit_tool):
        result = edit_tool.execute({"path": "../../etc/passwd", "old_text": "x", "new_text": "y"})
        assert not result.ok
        assert "access denied" in result.output.lower()

    def test_absolute_path_outside_project_rejected(self, edit_tool):
        result = edit_tool.execute({"path": "/etc/passwd", "old_text": "x", "new_text": "y"})
        assert not result.ok
        assert "access denied" in result.output.lower()

    def test_sensitive_file_rejected(self, edit_tool):
        result = edit_tool.execute(
            {"path": "id_rsa", "old_text": "shh", "new_text": "different"}
        )
        assert not result.ok
        assert "sensitive" in result.output.lower()

    def test_env_file_is_allowed(self, edit_tool):
        """.env is deliberately not treated as sensitive for editing (by
        request) -- unlike id_rsa/*.pem/etc above, it can be created and
        edited directly. See agent/project.py's ENV_FILE_PATTERNS."""
        result = edit_tool.execute({"path": ".env", "old_text": "SECRET=shh", "new_text": "SECRET=x"})
        assert result.ok
        assert result.pending_change is not None

    def test_directory_target_rejected(self, edit_tool):
        result = edit_tool.execute({"path": "backend", "old_text": "x", "new_text": "y"})
        assert not result.ok
        assert "directory" in result.output.lower()

    def test_identical_old_and_new_text_rejected(self, edit_tool):
        result = edit_tool.execute(
            {"path": "backend/auth.py", "old_text": "return None", "new_text": "return None"}
        )
        assert not result.ok
        assert "identical" in result.output.lower()

    def test_empty_old_text_rejected(self, edit_tool):
        result = edit_tool.execute({"path": "backend/auth.py", "old_text": "", "new_text": "y"})
        assert not result.ok

    def test_malformed_arguments_rejected(self, edit_tool):
        result = edit_tool.execute({"path": "backend/auth.py"})  # missing old_text/new_text
        assert not result.ok
        assert "invalid arguments" in result.output.lower()

    def test_stale_file_is_refused(self, project, tracker):
        target = project.root / "backend" / "auth.py"
        # Simulate: model read the file earlier...
        tracker.record(target, "some completely different content the model actually saw\n")
        # ...but the file on disk doesn't match that anymore.
        edit_tool = build_edit_file_tool(project, tracker)

        result = edit_tool.execute(
            {"path": "backend/auth.py", "old_text": "return None", "new_text": "return {}"}
        )
        assert not result.ok
        assert "changed on disk" in result.output.lower()

    def test_fresh_file_matching_tracker_is_allowed(self, project, tracker):
        target = project.root / "backend" / "auth.py"
        current_content = target.read_text()
        tracker.record(target, current_content)
        edit_tool = build_edit_file_tool(project, tracker)

        result = edit_tool.execute(
            {"path": "backend/auth.py", "old_text": "return None", "new_text": "return {}"}
        )
        assert result.ok

    def test_no_tracker_skips_staleness_check(self, project):
        edit_tool = build_edit_file_tool(project, tracker=None)
        result = edit_tool.execute(
            {"path": "backend/auth.py", "old_text": "return None", "new_text": "return {}"}
        )
        assert result.ok


class TestWriteFilePropose:
    def test_valid_new_file_proposes_without_writing(self, write_tool, project):
        result = write_tool.execute({"path": "backend/config.py", "content": "DEBUG = True\n"})
        assert result.ok
        assert result.pending_change is not None
        assert result.pending_change.kind == "create"
        assert result.pending_change.new_content == "DEBUG = True\n"
        assert not (project.root / "backend" / "config.py").exists()

    def test_existing_file_is_rejected(self, write_tool):
        result = write_tool.execute({"path": "backend/auth.py", "content": "overwrite attempt"})
        assert not result.ok
        assert "already exists" in result.output.lower()

    def test_path_traversal_rejected(self, write_tool):
        result = write_tool.execute({"path": "../../evil.py", "content": "x"})
        assert not result.ok
        assert "access denied" in result.output.lower()

    def test_absolute_path_outside_project_rejected(self, write_tool):
        result = write_tool.execute({"path": "/tmp/evil.py", "content": "x"})
        assert not result.ok
        assert "access denied" in result.output.lower()

    def test_sensitive_filename_rejected(self, write_tool):
        result = write_tool.execute({"path": "backend/id_rsa", "content": "fake key"})
        assert not result.ok
        assert "sensitive" in result.output.lower()

    def test_new_env_file_is_allowed(self, write_tool):
        result = write_tool.execute({"path": "backend/.env", "content": "API_KEY=x\n"})
        assert result.ok
        assert result.pending_change is not None

    def test_directory_target_rejected(self, write_tool):
        result = write_tool.execute({"path": "backend", "content": "x"})
        assert not result.ok

    def test_malformed_arguments_rejected(self, write_tool):
        result = write_tool.execute({"path": "backend/new.py"})  # missing content
        assert not result.ok


class TestApplyChange:
    def test_approved_edit_writes_and_verifies(self, project, tracker):
        edit_tool = build_edit_file_tool(project, tracker)
        propose = edit_tool.execute(
            {"path": "backend/auth.py", "old_text": "return None", "new_text": "return {'ok': False}"}
        )
        assert propose.ok

        apply_result = apply_change(propose.pending_change, tracker)

        assert apply_result.ok
        assert "updated" in apply_result.output.lower()
        on_disk = (project.root / "backend" / "auth.py").read_text()
        assert "return {'ok': False}" in on_disk
        assert "return None" not in on_disk

    def test_apply_updates_tracker_so_followup_edit_is_fresh(self, project, tracker):
        target = project.root / "backend" / "auth.py"
        tracker.record(target, target.read_text())
        edit_tool = build_edit_file_tool(project, tracker)

        propose1 = edit_tool.execute(
            {"path": "backend/auth.py", "old_text": "return None", "new_text": "return {}"}
        )
        apply_change(propose1.pending_change, tracker)

        # A second edit against the now-current content should not be "stale".
        propose2 = edit_tool.execute(
            {"path": "backend/auth.py", "old_text": "return {}", "new_text": "return {'x': 1}"}
        )
        assert propose2.ok

    def test_approved_write_creates_file(self, project, tracker):
        write_tool = build_write_file_tool(project, tracker)
        propose = write_tool.execute({"path": "backend/config.py", "content": "DEBUG = True\n"})
        assert propose.ok

        apply_result = apply_change(propose.pending_change, tracker)

        assert apply_result.ok
        assert "created" in apply_result.output.lower()
        assert (project.root / "backend" / "config.py").read_text() == "DEBUG = True\n"

    def test_apply_creates_missing_parent_directories(self, project, tracker):
        write_tool = build_write_file_tool(project, tracker)
        propose = write_tool.execute({"path": "backend/utils/validators.py", "content": "x = 1\n"})
        apply_result = apply_change(propose.pending_change, tracker)
        assert apply_result.ok
        assert (project.root / "backend" / "utils" / "validators.py").read_text() == "x = 1\n"

    def test_apply_preserves_permissions_on_edit(self, project, tracker):
        target = project.root / "backend" / "auth.py"
        os.chmod(target, 0o640)
        edit_tool = build_edit_file_tool(project, tracker)
        propose = edit_tool.execute(
            {"path": "backend/auth.py", "old_text": "return None", "new_text": "return {}"}
        )
        apply_change(propose.pending_change, tracker)
        mode = os.stat(target).st_mode & 0o777
        assert mode == 0o640

    def test_apply_preserves_crlf_line_endings(self, project, tracker):
        crlf_file = project.root / "backend" / "windows.py"
        crlf_file.write_bytes(b"a = 1\r\nb = 2\r\n")
        edit_tool = build_edit_file_tool(project, tracker)
        propose = edit_tool.execute(
            {"path": "backend/windows.py", "old_text": "a = 1", "new_text": "a = 100"}
        )
        assert propose.ok
        apply_change(propose.pending_change, tracker)
        raw = crlf_file.read_bytes()
        assert b"\r\n" in raw
        assert b"a = 100\r\n" in raw

    def test_failed_write_leaves_original_file_intact(self, project, tracker):
        edit_tool = build_edit_file_tool(project, tracker)
        propose = edit_tool.execute(
            {"path": "backend/auth.py", "old_text": "return None", "new_text": "return {}"}
        )
        assert propose.ok
        original_bytes = (project.root / "backend" / "auth.py").read_bytes()

        with mock.patch("agent.tools.editing.os.replace", side_effect=OSError("disk full")):
            apply_result = apply_change(propose.pending_change, tracker)

        assert not apply_result.ok
        assert "failed to write" in apply_result.output.lower()
        # Original file must be untouched, and no stray temp files left behind.
        assert (project.root / "backend" / "auth.py").read_bytes() == original_bytes
        leftovers = list((project.root / "backend").glob(".auth.py.*.tmp"))
        assert leftovers == []

    def test_verification_mismatch_is_reported(self, project, tracker):
        edit_tool = build_edit_file_tool(project, tracker)
        propose = edit_tool.execute(
            {"path": "backend/auth.py", "old_text": "return None", "new_text": "return {}"}
        )
        assert propose.ok

        # Simulate the on-disk read-back not matching what was written.
        with mock.patch(
            "agent.tools.editing._decode_and_detect_newline", return_value=("something else", "\n")
        ):
            apply_result = apply_change(propose.pending_change, tracker)

        assert not apply_result.ok
        assert "doesn't match" in apply_result.output.lower()
