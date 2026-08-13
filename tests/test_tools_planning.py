"""Tests for the create_plan/update_plan/get_plan tools in agent/tools/planning.py."""
from __future__ import annotations

import pytest

from agent.task_state import TaskState
from agent.tools.planning import build_create_plan_tool, build_get_plan_tool, build_update_plan_tool


@pytest.fixture
def task_state():
    return TaskState()


@pytest.fixture
def create_plan_tool():
    return build_create_plan_tool()


@pytest.fixture
def update_plan_tool(task_state):
    return build_update_plan_tool(task_state)


@pytest.fixture
def get_plan_tool(task_state):
    return build_get_plan_tool(task_state)


class TestCreatePlan:
    def test_valid_plan_is_proposed(self, create_plan_tool):
        result = create_plan_tool.execute(
            {"goal": "Add JWT authentication", "steps": ["Inspect auth", "Add endpoint", "Add tests"]}
        )
        assert result.ok
        assert result.pending_plan is not None
        assert result.pending_plan.goal == "Add JWT authentication"
        assert len(result.pending_plan.steps) == 3
        assert result.pending_plan.steps[0].id == 1
        assert all(s.status == "pending" for s in result.pending_plan.steps)

    def test_never_writes_to_task_state_directly(self, create_plan_tool):
        """create_plan only proposes -- adopting it into task_state happens
        in loop.py after approval, never inside the tool itself."""
        result = create_plan_tool.execute({"goal": "Add X", "steps": ["a", "b"]})
        assert result.ok
        # No task_state is even passed to this tool -- if it tried to adopt
        # state directly this constructor call would need one and fail.

    def test_empty_goal_rejected(self, create_plan_tool):
        result = create_plan_tool.execute({"goal": "   ", "steps": ["a", "b"]})
        assert not result.ok
        assert result.pending_plan is None

    def test_single_step_rejected(self, create_plan_tool):
        result = create_plan_tool.execute({"goal": "Add X", "steps": ["only one step"]})
        assert not result.ok
        assert "at least" in result.output.lower()

    def test_too_many_steps_rejected(self, create_plan_tool):
        result = create_plan_tool.execute({"goal": "Add X", "steps": [f"step {i}" for i in range(15)]})
        assert not result.ok

    def test_blank_steps_filtered_out(self, create_plan_tool):
        result = create_plan_tool.execute({"goal": "Add X", "steps": ["a", "  ", "b", ""]})
        assert result.ok
        assert len(result.pending_plan.steps) == 2

    def test_malformed_arguments_rejected(self, create_plan_tool):
        result = create_plan_tool.execute({"goal": "Add X"})  # missing steps
        assert not result.ok


class TestUpdatePlan:
    def _seed_plan(self, task_state):
        from agent.planner import Plan, PlanStep

        task_state.plan = Plan(
            goal="Add JWT auth",
            steps=[
                PlanStep(id=1, description="Inspect auth"),
                PlanStep(id=2, description="Add endpoint"),
            ],
        )

    def test_updates_step_status(self, task_state, update_plan_tool):
        self._seed_plan(task_state)
        result = update_plan_tool.execute({"step_id": 1, "status": "completed"})
        assert result.ok
        assert task_state.plan.get_step(1).status == "completed"

    def test_updates_note_for_blocked(self, task_state, update_plan_tool):
        self._seed_plan(task_state)
        result = update_plan_tool.execute(
            {"step_id": 2, "status": "blocked", "note": "missing SMTP credentials"}
        )
        assert result.ok
        assert task_state.plan.get_step(2).note == "missing SMTP credentials"

    def test_no_plan_yet_rejected(self, update_plan_tool):
        result = update_plan_tool.execute({"step_id": 1, "status": "completed"})
        assert not result.ok
        assert "no plan" in result.output.lower()

    def test_invalid_status_rejected(self, task_state, update_plan_tool):
        self._seed_plan(task_state)
        result = update_plan_tool.execute({"step_id": 1, "status": "done"})
        assert not result.ok

    def test_unknown_step_id_rejected(self, task_state, update_plan_tool):
        self._seed_plan(task_state)
        result = update_plan_tool.execute({"step_id": 99, "status": "completed"})
        assert not result.ok

    def test_output_reflects_updated_plan(self, task_state, update_plan_tool):
        self._seed_plan(task_state)
        result = update_plan_tool.execute({"step_id": 1, "status": "in_progress"})
        assert "Inspect auth" in result.output


class TestGetPlan:
    def test_no_plan_message(self, get_plan_tool):
        result = get_plan_tool.execute({})
        assert result.ok
        assert "no plan" in result.output.lower()

    def test_shows_current_plan(self, task_state, get_plan_tool):
        from agent.planner import Plan, PlanStep

        task_state.plan = Plan(goal="Add JWT auth", steps=[PlanStep(id=1, description="Inspect auth")])
        result = get_plan_tool.execute({})
        assert result.ok
        assert "Add JWT auth" in result.output
        assert "Inspect auth" in result.output
