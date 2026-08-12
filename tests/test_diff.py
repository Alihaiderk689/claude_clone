"""Tests for unified diff generation in agent/diff.py."""
from __future__ import annotations

from pathlib import Path

from agent.diff import ProposedChange, unified_diff_text


def test_edit_diff_shows_removed_and_added_lines():
    change = ProposedChange(
        path="backend/auth.py",
        resolved_path=Path("/tmp/backend/auth.py"),
        kind="edit",
        old_content="def login():\n    return None\n",
        new_content="def login():\n    return {'error': 'Invalid credentials'}\n",
    )
    diff_text = unified_diff_text(change)

    assert "-    return None" in diff_text
    assert "+    return {'error': 'Invalid credentials'}" in diff_text
    assert " def login():" in diff_text  # unchanged context line


def test_edit_diff_file_headers_reference_both_sides():
    change = ProposedChange(
        path="backend/auth.py",
        resolved_path=Path("/tmp/backend/auth.py"),
        kind="edit",
        old_content="a\n",
        new_content="b\n",
    )
    diff_text = unified_diff_text(change)

    assert "--- a/backend/auth.py" in diff_text
    assert "+++ b/backend/auth.py" in diff_text


def test_new_file_diff_is_all_additions_from_dev_null():
    change = ProposedChange(
        path="backend/config.py",
        resolved_path=Path("/tmp/backend/config.py"),
        kind="create",
        old_content=None,
        new_content="DEBUG = True\n",
    )
    diff_text = unified_diff_text(change)

    assert "--- /dev/null" in diff_text
    assert "+++ b/backend/config.py" in diff_text
    assert "+DEBUG = True" in diff_text
    assert "-DEBUG" not in diff_text


def test_no_op_change_produces_empty_diff():
    change = ProposedChange(
        path="x.py",
        resolved_path=Path("/tmp/x.py"),
        kind="edit",
        old_content="same\n",
        new_content="same\n",
    )
    assert unified_diff_text(change) == ""
