"""Short-term memory for the CURRENT task only.

TaskState holds a concise, bounded record of what the agent has already
done -- not full file contents, not raw terminal output, not the whole
conversation (see agent/context_manager.py for how tool results get turned
into these compact entries, and how they get compacted back into the
conversation the model actually sees). It lives only as long as the running
process; there is no persistence across sessions, and `/new` in cli.py
resets it to a blank TaskState with no effect on project files or Git
history.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import List, Optional

from .planner import Plan


def _new_task_id() -> str:
    return uuid.uuid4().hex[:12]

# Keeps summarize() bounded regardless of how long a task runs -- only the
# most recent entries of each kind are kept for display, matching Phase 6's
# "concise task memory, not a full log" requirement.
MAX_FILES_IN_SUMMARY = 10
MAX_COMMANDS_IN_SUMMARY = 5
MAX_ERRORS_IN_SUMMARY = 3

# Phase 9: bounds the underlying *storage* too, not just summarize()'s
# display slice -- without this, a long-running task's lists (especially
# commands_executed/errors_encountered/git_operations, which only ever grow)
# would hold an unbounded number of Python objects for the life of the
# process even though only the last few of each are ever shown. Comfortably
# larger than any of the MAX_*_IN_SUMMARY caps so nothing display-relevant
# is ever lost before it would have scrolled out of the summary anyway.
_MAX_STORED = 50


def _trim(items: list) -> None:
    if len(items) > _MAX_STORED:
        del items[: len(items) - _MAX_STORED]


@dataclass
class CommandRecord:
    display: str  # e.g. "pytest -v"
    exit_code: Optional[int]
    outcome: str  # short human-readable result, e.g. "18 passed, 2 failed"
    is_test: bool = False


@dataclass
class TaskState:
    # A fresh, short, process-local identifier for this task -- not
    # persisted anywhere, just useful for correlating log lines and (in the
    # HTTP server) /task/status responses across a task's lifetime. A new
    # TaskState (e.g. via /new or /task/new) gets a new one.
    task_id: str = field(default_factory=_new_task_id)
    goal: Optional[str] = None
    plan: Optional[Plan] = None

    # Ordered, deduplicated: most-recently-touched path moves to the end.
    files_inspected: List[str] = field(default_factory=list)
    files_modified: List[str] = field(default_factory=list)

    commands_executed: List[CommandRecord] = field(default_factory=list)
    errors_encountered: List[str] = field(default_factory=list)
    git_operations: List[str] = field(default_factory=list)

    def note_file_inspected(self, path: str) -> None:
        if path in self.files_inspected:
            self.files_inspected.remove(path)
        self.files_inspected.append(path)
        _trim(self.files_inspected)

    def note_file_modified(self, path: str) -> None:
        if path not in self.files_modified:
            self.files_modified.append(path)
            _trim(self.files_modified)
        # A file that changed needs to be re-read before it can be trusted
        # again -- drop any "already inspected" record for it (see
        # context_manager.py, which also invalidates the raw conversation
        # content this mirrors).
        if path in self.files_inspected:
            self.files_inspected.remove(path)

    def note_file_removed(self, path: str) -> None:
        """A file that no longer exists (deleted, or renamed away from this
        path) can't be trusted as "inspected" or "modified" anymore -- drop
        both records for it, same rationale as note_file_modified's
        files_inspected invalidation."""
        if path in self.files_inspected:
            self.files_inspected.remove(path)
        if path in self.files_modified:
            self.files_modified.remove(path)

    def note_command(self, record: CommandRecord) -> None:
        self.commands_executed.append(record)
        _trim(self.commands_executed)

    def note_error(self, message: str) -> None:
        self.errors_encountered.append(message)
        _trim(self.errors_encountered)

    def note_git_operation(self, message: str) -> None:
        self.git_operations.append(message)
        _trim(self.git_operations)

    def has_content(self) -> bool:
        return bool(
            self.goal
            or self.plan
            or self.files_inspected
            or self.files_modified
            or self.commands_executed
            or self.errors_encountered
            or self.git_operations
        )

    def summarize(self) -> str:
        """A compact, bounded block of the most important task state --
        never full file contents, never full terminal output. Injected into
        the system prompt each turn by context_manager.refresh_system_prompt
        instead of accumulating in the conversation history.
        """
        if not self.has_content():
            return ""

        lines: List[str] = []

        if self.goal:
            lines.append(f"Current task: {self.goal}")

        if self.plan:
            done = self.plan.completed_count()
            total = len(self.plan.steps)
            lines.append(f"Plan progress: {done}/{total} steps completed.")
            current = self.plan.current_step()
            if current:
                lines.append(f"Current step: {current.id}. {current.description}")
            if self.plan.is_blocked():
                for step in self.plan.steps:
                    if step.status in ("blocked", "failed"):
                        reason = f" — {step.note}" if step.note else ""
                        lines.append(f"Blocked step: {step.id}. {step.description}{reason}")

        if self.files_inspected:
            shown = self.files_inspected[-MAX_FILES_IN_SUMMARY:]
            lines.append(f"Files inspected: {', '.join(shown)}")

        if self.files_modified:
            lines.append(f"Files modified this task: {', '.join(self.files_modified)}")

        if self.commands_executed:
            lines.append("Recent commands:")
            for record in self.commands_executed[-MAX_COMMANDS_IN_SUMMARY:]:
                lines.append(f"  {record.display} -> {record.outcome}")

        if self.errors_encountered:
            lines.append("Unresolved errors:")
            for err in self.errors_encountered[-MAX_ERRORS_IN_SUMMARY:]:
                lines.append(f"  {err}")

        if self.git_operations:
            lines.append(f"Git actions taken: {'; '.join(self.git_operations[-MAX_COMMANDS_IN_SUMMARY:])}")

        return "\n".join(lines)
