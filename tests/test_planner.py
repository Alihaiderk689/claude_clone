"""Tests for the plain-data plan representation in agent/planner.py."""
from __future__ import annotations

from agent.planner import Plan, PlanStep


def make_plan(*statuses):
    return Plan(
        goal="Add JWT authentication",
        steps=[PlanStep(id=i + 1, description=f"Step {i + 1}", status=s) for i, s in enumerate(statuses)],
    )


class TestPlanStepLookup:
    def test_get_step_found(self):
        plan = make_plan("pending", "pending")
        step = plan.get_step(2)
        assert step is not None
        assert step.description == "Step 2"

    def test_get_step_not_found(self):
        plan = make_plan("pending")
        assert plan.get_step(99) is None


class TestCurrentStep:
    def test_first_in_progress_wins(self):
        plan = make_plan("completed", "in_progress", "pending")
        assert plan.current_step().id == 2

    def test_falls_back_to_first_pending(self):
        plan = make_plan("completed", "completed", "pending", "pending")
        assert plan.current_step().id == 3

    def test_none_when_all_completed(self):
        plan = make_plan("completed", "completed")
        assert plan.current_step() is None

    def test_none_when_blocked_with_no_pending_or_in_progress(self):
        plan = make_plan("completed", "blocked")
        assert plan.current_step() is None


class TestCompletionAndBlocking:
    def test_is_complete_true(self):
        plan = make_plan("completed", "completed")
        assert plan.is_complete() is True

    def test_is_complete_false_with_pending(self):
        plan = make_plan("completed", "pending")
        assert plan.is_complete() is False

    def test_is_complete_false_when_empty(self):
        plan = Plan(goal="x", steps=[])
        assert plan.is_complete() is False

    def test_is_blocked_true_for_blocked(self):
        plan = make_plan("completed", "blocked")
        assert plan.is_blocked() is True

    def test_is_blocked_true_for_failed(self):
        plan = make_plan("failed", "pending")
        assert plan.is_blocked() is True

    def test_is_blocked_false_normally(self):
        plan = make_plan("completed", "in_progress", "pending")
        assert plan.is_blocked() is False

    def test_completed_count(self):
        plan = make_plan("completed", "completed", "pending", "in_progress")
        assert plan.completed_count() == 2


class TestRenderLines:
    def test_markers_match_status(self):
        plan = make_plan("completed", "in_progress", "pending", "blocked")
        lines = plan.render_lines()
        assert lines[0].startswith("✓ 1.")
        assert lines[1].startswith("→ 2.")
        assert lines[2].startswith("○ 3.")
        assert lines[3].startswith("✗ 4.")

    def test_note_shown_for_blocked_step(self):
        plan = make_plan("blocked")
        plan.steps[0].note = "missing SMTP credentials"
        lines = plan.render_lines()
        assert "missing SMTP credentials" in lines[0]

    def test_note_not_shown_for_pending_step(self):
        plan = make_plan("pending")
        plan.steps[0].note = "should not appear"
        lines = plan.render_lines()
        assert "should not appear" not in lines[0]

    def test_line_count_matches_step_count(self):
        plan = make_plan("pending", "pending", "completed")
        assert len(plan.render_lines()) == 3
