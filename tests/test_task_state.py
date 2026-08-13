"""Tests for the short-term, current-task-only memory in agent/task_state.py."""
from __future__ import annotations

from agent.planner import Plan, PlanStep
from agent.task_state import CommandRecord, TaskState


class TestFileTracking:
    def test_note_file_inspected_adds_path(self):
        ts = TaskState()
        ts.note_file_inspected("backend/auth.py")
        assert ts.files_inspected == ["backend/auth.py"]

    def test_note_file_inspected_moves_repeat_to_end(self):
        ts = TaskState()
        ts.note_file_inspected("a.py")
        ts.note_file_inspected("b.py")
        ts.note_file_inspected("a.py")
        assert ts.files_inspected == ["b.py", "a.py"]

    def test_note_file_modified_adds_to_modified(self):
        ts = TaskState()
        ts.note_file_modified("backend/auth.py")
        assert ts.files_modified == ["backend/auth.py"]

    def test_note_file_modified_does_not_duplicate(self):
        ts = TaskState()
        ts.note_file_modified("backend/auth.py")
        ts.note_file_modified("backend/auth.py")
        assert ts.files_modified == ["backend/auth.py"]

    def test_note_file_modified_invalidates_inspected_cache(self):
        """This is the cache-invalidation requirement: a modified file must
        no longer be treated as an already-inspected, still-current file."""
        ts = TaskState()
        ts.note_file_inspected("backend/auth.py")
        assert "backend/auth.py" in ts.files_inspected

        ts.note_file_modified("backend/auth.py")
        assert "backend/auth.py" not in ts.files_inspected
        assert "backend/auth.py" in ts.files_modified


class TestCommandAndErrorTracking:
    def test_note_command_appends(self):
        ts = TaskState()
        ts.note_command(CommandRecord(display="pytest", exit_code=0, outcome="2 passed"))
        assert len(ts.commands_executed) == 1
        assert ts.commands_executed[0].outcome == "2 passed"

    def test_note_error_appends(self):
        ts = TaskState()
        ts.note_error("pytest failed: 1 failed, 2 passed")
        assert ts.errors_encountered == ["pytest failed: 1 failed, 2 passed"]

    def test_note_git_operation_appends(self):
        ts = TaskState()
        ts.note_git_operation("Created commit abc123: Fix bug")
        assert ts.git_operations == ["Created commit abc123: Fix bug"]


class TestHasContent:
    def test_false_for_fresh_state(self):
        assert TaskState().has_content() is False

    def test_true_after_goal_set(self):
        ts = TaskState()
        ts.goal = "Add JWT auth"
        assert ts.has_content() is True

    def test_true_after_file_inspected(self):
        ts = TaskState()
        ts.note_file_inspected("a.py")
        assert ts.has_content() is True


class TestSummarize:
    def test_empty_state_summarizes_to_empty_string(self):
        assert TaskState().summarize() == ""

    def test_includes_goal(self):
        ts = TaskState()
        ts.goal = "Add JWT authentication"
        assert "Add JWT authentication" in ts.summarize()

    def test_includes_files_inspected(self):
        ts = TaskState()
        ts.note_file_inspected("backend/auth.py")
        ts.note_file_inspected("backend/models.py")
        summary = ts.summarize()
        assert "backend/auth.py" in summary
        assert "backend/models.py" in summary

    def test_includes_files_modified(self):
        ts = TaskState()
        ts.note_file_modified("backend/auth.py")
        assert "backend/auth.py" in ts.summarize()

    def test_includes_recent_commands(self):
        ts = TaskState()
        ts.note_command(CommandRecord(display="pytest", exit_code=1, outcome="1 failed, 2 passed"))
        summary = ts.summarize()
        assert "pytest" in summary
        assert "1 failed, 2 passed" in summary

    def test_includes_errors(self):
        ts = TaskState()
        ts.note_error("token expiration comparison used local time incorrectly")
        assert "token expiration comparison used local time incorrectly" in ts.summarize()

    def test_includes_plan_progress(self):
        ts = TaskState()
        ts.plan = Plan(
            goal="x",
            steps=[
                PlanStep(id=1, description="Inspect auth", status="completed"),
                PlanStep(id=2, description="Add endpoint", status="in_progress"),
                PlanStep(id=3, description="Add tests", status="pending"),
            ],
        )
        summary = ts.summarize()
        assert "1/3 steps completed" in summary
        assert "Add endpoint" in summary

    def test_includes_blocked_reason(self):
        ts = TaskState()
        ts.plan = Plan(
            goal="x",
            steps=[PlanStep(id=1, description="Send reset email", status="blocked", note="no SMTP creds")],
        )
        summary = ts.summarize()
        assert "no SMTP creds" in summary

    def test_file_list_bounded_in_summary(self):
        ts = TaskState()
        for i in range(30):
            ts.note_file_inspected(f"file_{i}.py")
        summary = ts.summarize()
        # Only the most recent files should appear -- not all 30.
        assert "file_29.py" in summary
        assert "file_0.py" not in summary

    def test_command_list_bounded_in_summary(self):
        ts = TaskState()
        for i in range(20):
            ts.note_command(CommandRecord(display=f"cmd{i}", exit_code=0, outcome="ok"))
        summary = ts.summarize()
        assert "cmd19" in summary
        assert "cmd0" not in summary

    def test_does_not_include_raw_file_contents(self):
        """TaskState must never be handed full file contents to begin
        with -- this documents the contract that record_read_file only
        ever passes a bare path, never file bytes."""
        ts = TaskState()
        ts.note_file_inspected("backend/auth.py")
        summary = ts.summarize()
        assert len(summary) < 500  # a summary, not a dump
