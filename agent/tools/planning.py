"""Planning tools: create_plan, update_plan, get_plan.

create_plan follows the same propose/apply split as edit_file, run_command,
and the Git tools (see base.py's pending_plan) -- but with one deliberate
difference. A plan has no side effects of its own: it doesn't touch a file,
run anything, or change Git state, so its confirmation prompt defaults to
approve on a bare Enter ([Y/n], handled in cli.py) instead of the
default-reject every action tool uses. Approving a plan is never permission
to skip the file/command/Git approvals those actions still each require on
their own -- see agent/loop.py and cli.py.

update_plan/get_plan operate on a plan that's already been approved and
adopted into the task state; neither has any side effect of its own, so
neither needs approval -- they behave like any other read-only tool.
"""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field

from ..planner import VALID_STATUSES, Plan, PlanStep
from ..task_state import TaskState
from .base import Tool, ToolError, ToolResult

MIN_PLAN_STEPS = 2
MAX_PLAN_STEPS = 12


class CreatePlanArgs(BaseModel):
    goal: str = Field(description="A short, one-sentence description of the overall task.")
    steps: List[str] = Field(
        description=f"{MIN_PLAN_STEPS}-{MAX_PLAN_STEPS} short, concrete step descriptions, in order."
    )


class UpdatePlanArgs(BaseModel):
    step_id: int = Field(description="The step number to update.")
    status: str = Field(description="One of: pending, in_progress, completed, blocked, failed.")
    note: Optional[str] = Field(
        default=None, description="Short reason for the status, especially for blocked/failed."
    )


class GetPlanArgs(BaseModel):
    pass


def _create_plan(args: CreatePlanArgs) -> ToolResult:
    goal = args.goal.strip()
    if not goal:
        raise ToolError("goal must not be empty.")

    steps = [s.strip() for s in args.steps if s.strip()]
    if len(steps) < MIN_PLAN_STEPS:
        raise ToolError(
            f"A plan needs at least {MIN_PLAN_STEPS} steps. For a small, single-action request, "
            "don't call create_plan at all -- just do it directly."
        )
    if len(steps) > MAX_PLAN_STEPS:
        raise ToolError(f"Too many steps (max {MAX_PLAN_STEPS}) -- keep the plan at a higher level.")

    plan = Plan(goal=goal, steps=[PlanStep(id=i + 1, description=d) for i, d in enumerate(steps)])
    return ToolResult(
        ok=True,
        output="(pending user approval)",
        display=f"create_plan({goal!r})",
        pending_plan=plan,
    )


def _update_plan(task_state: TaskState, args: UpdatePlanArgs) -> ToolResult:
    if task_state.plan is None:
        raise ToolError("No plan exists yet. Call create_plan first.")
    if args.status not in VALID_STATUSES:
        raise ToolError(f"Invalid status {args.status!r}. Use one of: {', '.join(sorted(VALID_STATUSES))}.")

    step = task_state.plan.get_step(args.step_id)
    if step is None:
        valid_ids = ", ".join(str(s.id) for s in task_state.plan.steps)
        raise ToolError(f"No step with id {args.step_id} in the current plan. Valid ids: {valid_ids}.")

    step.status = args.status
    step.note = args.note.strip() if args.note else None

    lines = task_state.plan.render_lines()
    return ToolResult(
        ok=True,
        output="Plan updated:\n" + "\n".join(lines),
        display=f"update_plan({args.step_id}, {args.status!r})",
    )


def _get_plan(task_state: TaskState, _args: GetPlanArgs) -> ToolResult:
    if task_state.plan is None:
        return ToolResult(ok=True, output="No plan exists for the current task.", display="get_plan()")

    lines = task_state.plan.render_lines()
    return ToolResult(
        ok=True, output=f"Goal: {task_state.plan.goal}\n\n" + "\n".join(lines), display="get_plan()"
    )


def build_create_plan_tool() -> Tool:
    return Tool(
        name="create_plan",
        description=(
            f"Propose a structured, numbered plan ({MIN_PLAN_STEPS}-{MAX_PLAN_STEPS} short steps) for "
            "a multi-step task -- only for tasks that genuinely span several files or concerns (e.g. "
            "'add JWT authentication'). Do NOT call this for small, single-action requests (e.g. "
            "'rename this variable', 'change this button text') -- just do those directly. This only "
            "proposes the plan: the user is shown it and must approve before it becomes the active "
            "plan. Once approved, track progress with update_plan as you complete each step."
        ),
        args_model=CreatePlanArgs,
        run=_create_plan,
    )


def build_update_plan_tool(task_state: TaskState) -> Tool:
    return Tool(
        name="update_plan",
        description=(
            "Update the status of one step in the currently active plan (pending, in_progress, "
            "completed, blocked, failed). Mark a step in_progress before working on it and "
            "completed when it's genuinely done -- not just edited, but verified (tests passing, "
            "etc. as applicable). Use blocked/failed with a short note if you cannot continue a "
            "step; don't claim success instead."
        ),
        args_model=UpdatePlanArgs,
        run=lambda args: _update_plan(task_state, args),
    )


def build_get_plan_tool(task_state: TaskState) -> Tool:
    return Tool(
        name="get_plan",
        description="Show the current plan and each step's status. Useful to re-orient on a long task.",
        args_model=GetPlanArgs,
        run=lambda args: _get_plan(task_state, args),
    )
