"""Project-inspection and approval-gated editing tools available to the agent."""
from __future__ import annotations

from typing import Optional

from ..project import ProjectRoot
from ..task_state import TaskState
from .base import Tool, ToolError, ToolResult
from .editing import apply_change, build_edit_file_tool, build_write_file_tool
from .file_ops import apply_file_op, build_delete_file_tool, build_rename_file_tool
from .filesystem import build_list_files_tool, build_read_file_tool
from .git import (
    apply_git_operation,
    build_git_commit_tool,
    build_git_create_branch_tool,
    build_git_diff_tool,
    build_git_log_tool,
    build_git_push_tool,
    build_git_stage_tool,
    build_git_status_tool,
)
from .planning import build_create_plan_tool, build_get_plan_tool, build_update_plan_tool
from .registry import ToolRegistry
from .search import build_search_files_tool
from .state import FileStateTracker
from .terminal import build_run_command_tool, execute_command


def build_default_registry(
    project: ProjectRoot,
    tracker: Optional[FileStateTracker] = None,
    task_state: Optional[TaskState] = None,
) -> ToolRegistry:
    """Assemble the full toolset for a given project root.

    `tracker` is shared between read_file (which records what the model has
    seen) and edit_file (which checks it before proposing a change) so a
    stale edit can be caught. Pass the same tracker to run_agent_turn so
    apply_change() updates it after a write too — see cli.py's wiring.

    `task_state` backs the planning tools (update_plan/get_plan read and
    write the plan living on it; create_plan only proposes one, adopted by
    loop.py after approval) and is shared with agent/context_manager.py, so
    pass the same instance to run_agent_turn as well. read_file also reads
    (never writes) task_state.files_inspected, as the second, narrow
    exception to "tools don't know about TaskState" -- see the Phase 9
    unchanged-file short-circuit in tools/filesystem.py._read_file.

    Git tools are always registered, even outside a Git repository -- each
    one reports "This project is not a Git repository" gracefully rather
    than being conditionally hidden (see agent/tools/git.py).
    """
    tracker = tracker if tracker is not None else FileStateTracker()
    task_state = task_state if task_state is not None else TaskState()
    registry = ToolRegistry()
    registry.register(build_list_files_tool(project))
    registry.register(build_read_file_tool(project, tracker, task_state))
    registry.register(build_search_files_tool(project))
    registry.register(build_edit_file_tool(project, tracker))
    registry.register(build_write_file_tool(project, tracker))
    registry.register(build_delete_file_tool(project))
    registry.register(build_rename_file_tool(project))
    registry.register(build_run_command_tool(project))
    registry.register(build_git_status_tool(project))
    registry.register(build_git_diff_tool(project))
    registry.register(build_git_log_tool(project))
    registry.register(build_git_create_branch_tool(project))
    registry.register(build_git_stage_tool(project))
    registry.register(build_git_commit_tool(project))
    registry.register(build_git_push_tool(project))
    registry.register(build_create_plan_tool())
    registry.register(build_update_plan_tool(task_state))
    registry.register(build_get_plan_tool(task_state))
    return registry


__all__ = [
    "Tool",
    "ToolError",
    "ToolResult",
    "ToolRegistry",
    "FileStateTracker",
    "TaskState",
    "apply_change",
    "apply_file_op",
    "apply_git_operation",
    "execute_command",
    "build_default_registry",
]
