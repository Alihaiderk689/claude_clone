"""Structured plans for multi-step tasks.

A Plan is plain data (not prose) so the application -- not the model's
memory of the conversation -- is the source of truth for what's been done
and what's left. See agent/tools/planning.py for the tools that create and
update one, and agent/task_state.py for where the current plan lives.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

VALID_STATUSES = frozenset({"pending", "in_progress", "completed", "blocked", "failed"})

_STATUS_MARKERS = {
    "completed": "✓",
    "in_progress": "→",
    "blocked": "✗",
    "failed": "✗",
    "pending": "○",
}


@dataclass
class PlanStep:
    id: int
    description: str
    status: str = "pending"
    note: Optional[str] = None  # short reason, mainly used for blocked/failed


@dataclass
class Plan:
    goal: str
    steps: List[PlanStep] = field(default_factory=list)

    def get_step(self, step_id: int) -> Optional[PlanStep]:
        for step in self.steps:
            if step.id == step_id:
                return step
        return None

    def current_step(self) -> Optional[PlanStep]:
        """The step the agent should be working on: the first in_progress
        step, or failing that, the first pending one."""
        for step in self.steps:
            if step.status == "in_progress":
                return step
        for step in self.steps:
            if step.status == "pending":
                return step
        return None

    def is_complete(self) -> bool:
        return bool(self.steps) and all(s.status == "completed" for s in self.steps)

    def is_blocked(self) -> bool:
        return any(s.status in ("blocked", "failed") for s in self.steps)

    def completed_count(self) -> int:
        return sum(1 for s in self.steps if s.status == "completed")

    def render_lines(self) -> List[str]:
        lines = []
        for step in self.steps:
            marker = _STATUS_MARKERS.get(step.status, "○")
            line = f"{marker} {step.id}. {step.description}"
            if step.note and step.status in ("blocked", "failed"):
                line += f" ({step.note})"
            lines.append(line)
        return lines
