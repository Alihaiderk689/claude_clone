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

    def test_placeholder_old_text_rejected_with_a_directive_message(self, edit_tool):
        """Observed live: a small model, after already calling read_file
        successfully, sent old_text='<existing text to replace>' instead of
        real content copied from that read. This must fail with a message
        that points at read_file's output, not the generic 'not found'
        error a real (but wrong) old_text would get -- that generic message
        was observed live not being enough to redirect the model."""
        result = edit_tool.execute(
            {
                "path": "backend/auth.py",
                "old_text": "<existing text to replace>",
                "new_text": "y = 1",
            }
        )
        assert not result.ok
        assert "placeholder" in result.output.lower()
        assert "read_file" in result.output

    def test_placeholder_new_text_rejected(self, edit_tool):
        result = edit_tool.execute(
            {
                "path": "backend/auth.py",
                "old_text": "return None",
                "new_text": "<new content here>",
            }
        )
        assert not result.ok
        assert "placeholder" in result.output.lower()

    def test_real_old_text_containing_angle_brackets_is_still_allowed(self, project, tracker):
        """The placeholder check must be narrow enough that genuine code
        using '<'/'>' (comparisons, generics, HTML) is never blocked -- only
        an argument that is *entirely* one bracketed phrase is rejected."""
        target = project.root / "generic.py"
        target.write_text("def f(x):\n    return x < 10 and x > 0\n")
        tracker.record(target, target.read_text())
        edit_tool = build_edit_file_tool(project, tracker)

        result = edit_tool.execute(
            {
                "path": "generic.py",
                "old_text": "return x < 10 and x > 0",
                "new_text": "return 0 < x < 10",
            }
        )
        assert result.ok
        assert result.pending_change is not None

    def test_malformed_arguments_rejected(self, edit_tool):
        result = edit_tool.execute({"path": "backend/auth.py"})  # missing old_text/new_text
        assert not result.ok
        assert "invalid arguments" in result.output.lower()

    def test_edit_producing_invalid_python_indentation_is_rejected(self, project, tracker):
        """The exact live-observed failure: every nested line (function
        body, for loop, nested for loop, if statement) at the same 1-space
        indent instead of increasing per block -- a real IndentationError,
        not just unconventional style. Must be caught before it's ever
        proposed as an approvable diff, let alone written to disk."""
        target = project.root / "sorter.py"
        target.write_text("def bubble_sort(arr):\n    pass\n")
        tracker.record(target, target.read_text())
        edit_tool = build_edit_file_tool(project, tracker)

        broken_body = (
            "def bubble_sort(arr):\n"
            " n = len(arr)\n"
            " for i in range(n):\n"
            " for j in range(0, n-i-1):\n"
            " if arr[j] > arr[j+1]:\n"
            " arr[j], arr[j+1] = arr[j+1], arr[j]\n"
        )
        result = edit_tool.execute(
            {"path": "sorter.py", "old_text": "def bubble_sort(arr):\n    pass\n", "new_text": broken_body}
        )
        assert not result.ok
        assert result.pending_change is None
        assert "not valid python" in result.output.lower()
        assert "line" in result.output.lower()

    def test_edit_producing_valid_python_is_still_allowed(self, project, tracker):
        target = project.root / "sorter.py"
        target.write_text("def bubble_sort(arr):\n    pass\n")
        tracker.record(target, target.read_text())
        edit_tool = build_edit_file_tool(project, tracker)

        correct_body = (
            "def bubble_sort(arr):\n"
            "    n = len(arr)\n"
            "    for i in range(n):\n"
            "        for j in range(0, n - i - 1):\n"
            "            if arr[j] > arr[j + 1]:\n"
            "                arr[j], arr[j + 1] = arr[j + 1], arr[j]\n"
        )
        result = edit_tool.execute(
            {"path": "sorter.py", "old_text": "def bubble_sort(arr):\n    pass\n", "new_text": correct_body}
        )
        assert result.ok
        assert result.pending_change is not None

    def test_syntax_check_is_skipped_for_non_python_files(self, project, tracker):
        """The checker is Python-specific -- it must not reject content in
        any other file type just because it wouldn't parse as Python."""
        target = project.root / "notes.md"
        target.write_text("# Notes\nold line\n")
        tracker.record(target, target.read_text())
        edit_tool = build_edit_file_tool(project, tracker)

        result = edit_tool.execute(
            {"path": "notes.md", "old_text": "old line", "new_text": "def broken(:\n  not python"}
        )
        assert result.ok
        assert result.pending_change is not None

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

    def test_new_file_with_invalid_python_syntax_is_rejected(self, write_tool):
        result = write_tool.execute(
            {"path": "backend/broken.py", "content": "def f(:\n    return 1\n"}
        )
        assert not result.ok
        assert result.pending_change is None
        assert "not valid python" in result.output.lower()

    def test_new_non_python_file_with_python_like_garbage_is_still_allowed(self, write_tool):
        result = write_tool.execute({"path": "backend/notes.txt", "content": "def f(:\n  nonsense"})
        assert result.ok
        assert result.pending_change is not None


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


class TestLineNumberPrefixTolerance:
    """Fix 1: read_file renders 'NNN | code', but the file on disk has no such
    prefix. A model copying its own read_file output verbatim into old_text
    was the single largest source of "target text not found" failures."""

    def test_old_text_with_read_file_line_numbers_still_matches(self, edit_tool, project):
        result = edit_tool.execute(
            {
                "path": "backend/auth.py",
                "old_text": "    4 |     return None",
                "new_text": "    return {'error': 'bad'}",
            }
        )

        assert result.ok, result.output
        change = result.pending_change
        assert "return {'error': 'bad'}" in change.new_content
        # The prefix itself must never reach the file.
        assert "|" not in change.new_content
        assert "4 |" not in change.new_content

    def test_multiline_old_text_with_line_numbers_matches(self, edit_tool):
        result = edit_tool.execute(
            {
                "path": "backend/auth.py",
                "old_text": "    2 |     if username == 'admin' and password == '1234':\n"
                "    3 |         return {'token': 'abc123'}",
                "new_text": "    if check(username, password):\n        return issue_token()",
            }
        )

        assert result.ok, result.output
        assert "check(username, password)" in result.pending_change.new_content
        assert "abc123" not in result.pending_change.new_content

    def test_line_numbers_stripped_from_new_text_too(self, edit_tool):
        result = edit_tool.execute(
            {
                "path": "backend/auth.py",
                "old_text": "    4 |     return None",
                "new_text": "    4 |     return {}",
            }
        )

        assert result.ok, result.output
        assert "return {}" in result.pending_change.new_content
        assert "4 |" not in result.pending_change.new_content

    def test_identical_after_stripping_is_rejected(self, edit_tool):
        result = edit_tool.execute(
            {
                "path": "backend/auth.py",
                "old_text": "    4 |     return None",
                "new_text": "    4 |     return None",
            }
        )

        assert not result.ok
        assert "nothing to change" in result.output.lower()

    def test_real_content_that_merely_looks_numbered_is_not_mangled(self, project, tracker):
        """The strip only fires when a literal match already failed AND every
        non-blank line carries the prefix -- real pipe-containing code must
        keep working."""
        (project.root / "data.txt").write_text("1 | alpha\n2 | beta\n")
        tool = build_edit_file_tool(project, tracker)

        result = tool.execute(
            {"path": "data.txt", "old_text": "2 | beta", "new_text": "2 | gamma"}
        )

        assert result.ok, result.output
        assert result.pending_change.new_content == "1 | alpha\n2 | gamma\n"

    def test_exact_match_always_wins_over_stripping(self, project, tracker):
        (project.root / "mixed.txt").write_text("   7 | keep\nreal line\n")
        tool = build_edit_file_tool(project, tracker)

        result = tool.execute(
            {"path": "mixed.txt", "old_text": "   7 | keep", "new_text": "   7 | kept"}
        )

        assert result.ok, result.output
        assert result.pending_change.new_content == "   7 | kept\nreal line\n"


class TestWhitespaceInsensitiveFallback:
    """Fix 2: an edit whose only defect is indentation drift should succeed,
    with the file's own indentation re-applied to the replacement."""

    def test_wrong_base_indentation_still_matches_and_is_reindented(self, edit_tool):
        """The recoverable drift: relative nesting is right, absolute
        indentation is wrong (the file indents these 4 and 8 spaces)."""
        result = edit_tool.execute(
            {
                "path": "backend/auth.py",
                "old_text": "if username == 'admin' and password == '1234':\n"
                "    return {'token': 'abc123'}",
                "new_text": "if verify(username, password):\n    return issue_token()",
            }
        )

        assert result.ok, result.output
        new = result.pending_change.new_content
        assert "    if verify(username, password):\n" in new
        assert "        return issue_token()\n" in new

    def test_relative_nesting_is_preserved_when_reindenting(self, project, tracker):
        """A .txt fixture so _validate_python_syntax can't mask what the
        re-indent itself produced."""
        (project.root / "note.txt").write_text("root\n    alpha\n        beta\n")
        tool = build_edit_file_tool(project, tracker)

        result = tool.execute(
            {
                "path": "note.txt",
                # Nesting is right, base indent is wrong -> no literal match.
                "old_text": "alpha\n    beta",
                "new_text": "ALPHA\n    BETA\n        GAMMA",
            }
        )

        assert result.ok, result.output
        assert result.pending_change.new_content == (
            "root\n    ALPHA\n        BETA\n            GAMMA\n"
        )

    def test_reindent_anchors_on_the_shallowest_matched_line(self, project, tracker):
        (project.root / "deep.txt").write_text("root\n        deep\n    shallow\n")
        tool = build_edit_file_tool(project, tracker)

        result = tool.execute(
            {
                "path": "deep.txt",
                "old_text": "deep\nshallow",
                "new_text": "D\nS",
            }
        )

        assert result.ok, result.output
        # 4 (the shallowest matched line), not 8 (the first one).
        assert result.pending_change.new_content == "root\n    D\n    S\n"

    def test_fully_flattened_new_text_is_caught_by_the_syntax_check(self, edit_tool):
        """Nesting the model never supplied cannot be invented. The
        re-indent applies one base indent uniformly, and _validate_python_syntax
        refuses the result rather than writing broken Python."""
        result = edit_tool.execute(
            {
                "path": "backend/auth.py",
                "old_text": "if username == 'admin' and password == '1234':\n"
                "return {'token': 'abc123'}",
                "new_text": "if verify(username, password):\nreturn issue_token()",
            }
        )

        assert not result.ok
        assert "not valid python" in result.output.lower()
        assert result.pending_change is None

    def test_ambiguous_whitespace_match_is_rejected_without_replace_all(
        self, project, tracker
    ):
        (project.root / "twice.py").write_text("def a():\n    x = 1\n\ndef b():\n  x = 1\n")
        tool = build_edit_file_tool(project, tracker)

        result = tool.execute({"path": "twice.py", "old_text": "x = 1", "new_text": "x = 2"})

        assert not result.ok
        assert "ambiguous" in result.output.lower()

    def test_not_found_error_names_the_line_number_prefix_and_overwrite(self, edit_tool):
        result = edit_tool.execute(
            {"path": "backend/auth.py", "old_text": "nothing like this", "new_text": "x"}
        )

        assert not result.ok
        assert "NNN | " in result.output
        assert "overwrite=true" in result.output


class TestReplaceAll:
    """Fix 3: multiple occurrences report where they are, and replace_all
    turns the hard failure into an explicit opt-in."""

    def test_ambiguous_match_reports_line_numbers(self, edit_tool):
        result = edit_tool.execute(
            {"path": "backend/dup.py", "old_text": "x = 1", "new_text": "x = 2"}
        )

        assert not result.ok
        assert "2 times" in result.output
        assert "lines 1, 2" in result.output
        assert "replace_all=true" in result.output

    def test_replace_all_replaces_every_occurrence(self, edit_tool):
        result = edit_tool.execute(
            {
                "path": "backend/dup.py",
                "old_text": "x = 1",
                "new_text": "x = 2",
                "replace_all": True,
            }
        )

        assert result.ok, result.output
        assert result.pending_change.new_content == "x = 2\nx = 2\n"

    def test_replace_all_defaults_to_false(self, edit_tool):
        result = edit_tool.execute(
            {"path": "backend/auth.py", "old_text": "return None", "new_text": "return {}"}
        )
        assert result.ok
        assert result.pending_change.new_content.count("return {}") == 1

    def test_replace_all_works_through_the_whitespace_fallback(self, project, tracker):
        (project.root / "twice.py").write_text("def a():\n    x = 1\n\ndef b():\n  x = 1\n")
        tool = build_edit_file_tool(project, tracker)

        result = tool.execute(
            {"path": "twice.py", "old_text": "x = 1", "new_text": "x = 2", "replace_all": True}
        )

        assert result.ok, result.output
        # Each occurrence keeps its own original indentation.
        assert result.pending_change.new_content == "def a():\n    x = 2\n\ndef b():\n  x = 2\n"


class TestWriteFileOverwrite:
    """Fix 4: a single-approval full-file rewrite, for when edit_file's anchor
    matching keeps failing."""

    def test_existing_file_without_overwrite_still_fails(self, write_tool):
        result = write_tool.execute({"path": "backend/auth.py", "content": "x = 1\n"})

        assert not result.ok
        assert "already exists" in result.output
        assert "overwrite=true" in result.output

    def test_overwrite_proposes_an_edit_against_the_real_old_content(self, write_tool, project):
        original = (project.root / "backend" / "auth.py").read_text()

        result = write_tool.execute(
            {
                "path": "backend/auth.py",
                "content": "def login(u, p):\n    return None\n",
                "overwrite": True,
            }
        )

        assert result.ok, result.output
        change = result.pending_change
        # kind="edit" so the CLI shows a real diff and "Modified file".
        assert change.kind == "edit"
        assert change.old_content == original
        assert change.new_content == "def login(u, p):\n    return None\n"
        assert change.original_mode is not None
        # Still nothing written.
        assert (project.root / "backend" / "auth.py").read_text() == original

    def test_overwrite_still_validates_python_syntax(self, write_tool):
        result = write_tool.execute(
            {"path": "backend/auth.py", "content": "def broken(:\n", "overwrite": True}
        )

        assert not result.ok
        assert "not valid python" in result.output.lower()
        assert result.pending_change is None

    def test_overwrite_with_identical_content_is_rejected(self, write_tool, project):
        original = (project.root / "backend" / "auth.py").read_text()

        result = write_tool.execute(
            {"path": "backend/auth.py", "content": original, "overwrite": True}
        )

        assert not result.ok
        assert "nothing to change" in result.output.lower()

    def test_overwrite_still_refuses_a_sensitive_file(self, write_tool):
        result = write_tool.execute(
            {"path": "id_rsa", "content": "x\n", "overwrite": True}
        )

        assert not result.ok
        assert "sensitive" in result.output.lower()

    def test_overwrite_creates_a_new_file_when_none_exists(self, write_tool):
        result = write_tool.execute(
            {"path": "brand_new.py", "content": "x = 1\n", "overwrite": True}
        )

        assert result.ok, result.output
        assert result.pending_change.kind == "create"
        assert result.pending_change.old_content is None

    def test_approved_overwrite_replaces_the_file_on_disk(self, write_tool, project, tracker):
        result = write_tool.execute(
            {"path": "backend/auth.py", "content": "x = 1\n", "overwrite": True}
        )
        assert result.ok, result.output

        applied = apply_change(result.pending_change, tracker)

        assert applied.ok, applied.output
        assert (project.root / "backend" / "auth.py").read_text() == "x = 1\n"
