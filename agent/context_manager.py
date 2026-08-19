"""Turns tool results into concise task memory, and keeps the conversation
sent to Ollama from growing without bound.

Two separate concerns live here, matching Phase 6's diagram:

    Tool result -> extracted into TaskState (record_* functions)
    Full messages -> compacted before each LLM call (compact_messages)
    TaskState -> injected as a short summary (refresh_system_prompt)

Nothing here ever drops information the model actually needs for its
current step -- only content that's superseded (an old read of a file that
was re-read or edited since) or that's simply old and past a size budget,
with a compact summary of what it said always available in the system
prompt as a replacement. This matters concretely on 8GB of RAM with
qwen2.5-coder:3b: every character resent to Ollama costs real prompt-eval
time on every single turn, not just once.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Tuple

from .command_policy import ApprovedCommand
from .task_state import CommandRecord, TaskState
from .tools.terminal import CommandExecutionResult

# Conservative for a 3B model on 8GB unified memory -- every character here
# is prompt-eval cost paid again on every subsequent turn.
MAX_CONTEXT_CHARS = 12_000
# Never trim the most recent N tool messages, regardless of size -- the
# model needs its immediate context intact.
KEEP_RECENT_TOOL_MESSAGES = 6

TASK_MEMORY_MARKER = "\n\n---\nCurrent task memory:\n"

_SUPERSEDED_PLACEHOLDER = (
    "[earlier read_file result for {path} omitted -- a newer read of it follows "
    "later in this conversation; use that one instead]"
)
_STALE_PLACEHOLDER = (
    "[stale -- {path} was modified after this read; do not rely on this content, "
    "read_file it again if you need its current contents]"
)
_TRIMMED_PLACEHOLDER = "[tool output omitted to save context -- see the task memory summary above]"

_TEST_SUMMARY_RE = re.compile(r"(\d+\s+(?:passed|failed|error|errors|skipped)[\w\s,]*)", re.IGNORECASE)


@dataclass
class CompactionStats:
    """What compact_messages actually did on one call -- purely for debug
    logging/benchmarking (Phase 9); callers that ignore the return value are
    unaffected, same as before it existed."""

    superseded: int = 0
    stale: int = 0
    trimmed: int = 0

    def __bool__(self) -> bool:
        return bool(self.superseded or self.stale or self.trimmed)


# --------------------------------------------------------------------------
# Recording: tool results -> compact TaskState entries
# --------------------------------------------------------------------------


def record_read_file(task_state: Optional[TaskState], path: str) -> None:
    if task_state is not None and path:
        task_state.note_file_inspected(path)


def record_file_modified(task_state: Optional[TaskState], path: str) -> None:
    if task_state is not None and path:
        task_state.note_file_modified(path)


def record_file_removed(task_state: Optional[TaskState], path: str) -> None:
    if task_state is not None and path:
        task_state.note_file_removed(path)


def _looks_like_test_command(cmd: ApprovedCommand) -> bool:
    if cmd.program == "pytest":
        return True
    if cmd.program in ("python", "python3") and "pytest" in cmd.args:
        return True
    if cmd.program == "npm" and "test" in cmd.args:
        return True
    return False


def _summarize_command_outcome(exec_result: CommandExecutionResult) -> str:
    if exec_result.timed_out:
        return "timed out"
    tail = (exec_result.stdout or "")[-500:]
    match = None
    for match in _TEST_SUMMARY_RE.finditer(tail):
        pass  # keep the last match -- pytest's own summary line is at the end
    if match:
        return match.group(1).strip().strip("= ")
    if exec_result.exit_code == 0:
        return "succeeded (exit code 0)"
    return f"exit code {exec_result.exit_code}"


def _looks_fully_passing(outcome: str) -> bool:
    lowered = outcome.lower()
    if "failed" in lowered or "error" in lowered:
        return False
    return "passed" in lowered or "exit code 0" in lowered


def record_command_result(
    task_state: Optional[TaskState], cmd: ApprovedCommand, exec_result: CommandExecutionResult
) -> None:
    if task_state is None:
        return

    display = f"{cmd.program} {' '.join(cmd.args)}".strip()
    outcome = _summarize_command_outcome(exec_result)
    is_test = _looks_like_test_command(cmd)
    task_state.note_command(
        CommandRecord(display=display, exit_code=exec_result.exit_code, outcome=outcome, is_test=is_test)
    )

    if exec_result.timed_out:
        task_state.note_error(f"{display} timed out after {cmd.timeout}s")
    elif exec_result.exit_code not in (0, None):
        task_state.note_error(f"{display} failed: {outcome}")
    elif is_test and _looks_fully_passing(outcome):
        # A clean, fully-passing test run makes earlier recorded failures
        # stale -- clear them so the model doesn't keep citing a fixed bug.
        task_state.errors_encountered.clear()


def record_git_result(task_state: Optional[TaskState], kind: str, ok: bool, message: str) -> None:
    if task_state is not None and ok:
        task_state.note_git_operation(message)


# --------------------------------------------------------------------------
# Injecting compact memory into the system prompt
# --------------------------------------------------------------------------


def refresh_system_prompt(messages: List[dict], task_state: Optional[TaskState]) -> None:
    """Regenerates (not appends to) the task-memory section of the system
    prompt from the current TaskState, so it always reflects the latest
    state and never grows unboundedly on its own.
    """
    if not messages or messages[0].get("role") != "system":
        return
    base = messages[0]["content"].split(TASK_MEMORY_MARKER, 1)[0]
    summary = task_state.summarize() if task_state is not None else ""
    messages[0]["content"] = base + (TASK_MEMORY_MARKER + summary if summary else "")


# --------------------------------------------------------------------------
# Compacting the raw conversation before it's sent to Ollama
# --------------------------------------------------------------------------


def _iter_tool_messages_with_args(
    messages: List[dict],
) -> Iterator[Tuple[int, str, Dict[str, Any], dict]]:
    """Yields (index, tool_name, args, message) for every tool-result
    message, paired with the arguments that produced it by walking the
    preceding assistant tool_calls in order -- loop.py always appends tool
    results in the same order the corresponding calls were requested, so a
    simple FIFO queue correctly pairs each one back to its real arguments
    without depending on any particular tool's output format.
    """
    pending_calls: List[Tuple[str, Dict[str, Any]]] = []
    for i, msg in enumerate(messages):
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            for call in msg["tool_calls"]:
                function = call.get("function", {}) or {}
                name = function.get("name", "")
                raw_args = function.get("arguments")
                if isinstance(raw_args, str):
                    try:
                        raw_args = json.loads(raw_args)
                    except json.JSONDecodeError:
                        raw_args = {}
                pending_calls.append((name, raw_args or {}))
        elif msg.get("role") == "tool":
            if pending_calls:
                name, args = pending_calls.pop(0)
            else:
                name, args = msg.get("tool_name", ""), {}
            yield i, name, args, msg


def _replace_if_not_already_placeholder(msg: dict, placeholder: str) -> bool:
    content = msg.get("content") or ""
    if content and not content.startswith("["):
        msg["content"] = placeholder
        return True
    return False


def compact_messages(
    messages: List[dict],
    task_state: Optional[TaskState] = None,
    max_context_chars: int = MAX_CONTEXT_CHARS,
    keep_recent: int = KEEP_RECENT_TOOL_MESSAGES,
) -> CompactionStats:
    """Mutates `messages` in place. Two passes:

    1. Per-path read_file bookkeeping: an older read_file result for a path
       that was read again later is replaced with a short "superseded"
       placeholder; one that predates a later edit_file/write_file call for
       that same path is replaced with a "stale" placeholder instead, so the
       model is never left reasoning from content it no longer matches on
       disk. (edit_file's own stale-edit check in tools/editing.py is the
       real safety backstop for actually writing a file; this pass is about
       what the model *sees* in conversation history, not file-write safety.)
       A "newer" read_file result that's itself a Phase 9 cache-hit stub
       (`msg["cache_hit"]` set by loop.py -- see tools/filesystem.py's
       unchanged-file short-circuit) never supersedes anything: it carries
       no real content of its own, so the *older* real read is still the
       only actual copy of the file in context and must survive untouched.

    2. Only if the context is still large after that: trim the oldest tool
       output down to short placeholders, working from the earliest tool
       message forward and never touching the system prompt or the most
       recent `keep_recent` tool messages.

    Never touches non-tool messages (user/assistant/system content).

    Returns a CompactionStats describing what happened, for debug
    logging/benchmarking -- callers that ignore it are unaffected.
    """
    stats = CompactionStats()
    last_read_index: Dict[str, int] = {}
    for i, name, args, msg in _iter_tool_messages_with_args(messages):
        if name == "read_file":
            path = args.get("path")
            if not path:
                continue
            is_cache_hit_stub = bool(msg.get("cache_hit"))
            if path in last_read_index and not is_cache_hit_stub:
                if _replace_if_not_already_placeholder(
                    messages[last_read_index[path]], _SUPERSEDED_PLACEHOLDER.format(path=path)
                ):
                    stats.superseded += 1
            if not is_cache_hit_stub:
                last_read_index[path] = i
        elif name in ("edit_file", "write_file", "delete_file"):
            path = args.get("path")
            if path and path in last_read_index:
                if _replace_if_not_already_placeholder(
                    messages[last_read_index[path]], _STALE_PLACEHOLDER.format(path=path)
                ):
                    stats.stale += 1
                del last_read_index[path]
        elif name == "rename_file":
            # rename_file's args are source/destination, not path -- only the
            # earlier read of the SOURCE path is now stale (it no longer
            # exists there); the destination was never read under its new
            # name, so there's nothing to invalidate for it.
            source = args.get("source")
            if source and source in last_read_index:
                if _replace_if_not_already_placeholder(
                    messages[last_read_index[source]], _STALE_PLACEHOLDER.format(path=source)
                ):
                    stats.stale += 1
                del last_read_index[source]

    tool_indices = [i for i, m in enumerate(messages) if m.get("role") == "tool"]
    protected = set(tool_indices[-keep_recent:]) if keep_recent else set()

    def total_chars() -> int:
        return sum(len(m.get("content") or "") for m in messages)

    for i in tool_indices:
        if total_chars() <= max_context_chars:
            break
        if i in protected:
            continue
        content = messages[i].get("content") or ""
        if len(content) > len(_TRIMMED_PLACEHOLDER) and not content.startswith("["):
            messages[i]["content"] = _TRIMMED_PLACEHOLDER
            stats.trimmed += 1

    return stats
