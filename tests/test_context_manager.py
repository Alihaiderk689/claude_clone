"""Tests for agent/context_manager.py: turning tool results into compact
task memory, injecting that memory into the system prompt, and compacting
the raw conversation sent to Ollama.
"""
from __future__ import annotations

from pathlib import Path

from agent import context_manager
from agent.command_policy import ApprovedCommand
from agent.task_state import TaskState
from agent.tools.terminal import CommandExecutionResult


def make_cmd(program="pytest", args=None, timeout=120):
    return ApprovedCommand(program=program, args=args or [], timeout=timeout, cwd=Path("/tmp/repo"))


def make_result(exit_code=0, stdout="", stderr="", timed_out=False):
    return CommandExecutionResult(
        program="pytest", args=[], exit_code=exit_code, stdout=stdout, stderr=stderr, timed_out=timed_out
    )


class TestRecordReadFile:
    def test_records_path(self):
        ts = TaskState()
        context_manager.record_read_file(ts, "backend/auth.py")
        assert "backend/auth.py" in ts.files_inspected

    def test_none_task_state_is_a_safe_noop(self):
        context_manager.record_read_file(None, "backend/auth.py")  # must not raise


class TestRecordFileModified:
    def test_records_and_invalidates(self):
        ts = TaskState()
        context_manager.record_read_file(ts, "backend/auth.py")
        context_manager.record_file_modified(ts, "backend/auth.py")
        assert "backend/auth.py" in ts.files_modified
        assert "backend/auth.py" not in ts.files_inspected


class TestRecordCommandResult:
    def test_records_command(self):
        ts = TaskState()
        cmd = make_cmd(args=["-v"])
        result = make_result(exit_code=0, stdout="2 passed in 0.1s")
        context_manager.record_command_result(ts, cmd, result)
        assert len(ts.commands_executed) == 1
        assert ts.commands_executed[0].display == "pytest -v"
        assert "2 passed" in ts.commands_executed[0].outcome

    def test_failed_command_recorded_as_error(self):
        ts = TaskState()
        cmd = make_cmd()
        result = make_result(exit_code=1, stdout="1 failed, 2 passed in 0.1s")
        context_manager.record_command_result(ts, cmd, result)
        assert len(ts.errors_encountered) == 1
        assert "failed" in ts.errors_encountered[0].lower()

    def test_timeout_recorded_as_error(self):
        ts = TaskState()
        cmd = make_cmd(timeout=30)
        result = make_result(exit_code=None, timed_out=True)
        context_manager.record_command_result(ts, cmd, result)
        assert len(ts.errors_encountered) == 1
        assert "timed out" in ts.errors_encountered[0].lower()

    def test_successful_test_run_clears_prior_errors(self):
        ts = TaskState()
        ts.note_error("some stale earlier failure")
        cmd = make_cmd()
        result = make_result(exit_code=0, stdout="20 passed in 0.5s")
        context_manager.record_command_result(ts, cmd, result)
        assert ts.errors_encountered == []

    def test_successful_non_test_command_does_not_clear_errors(self):
        ts = TaskState()
        ts.note_error("some earlier failure")
        cmd = make_cmd(program="npm", args=["run", "lint"])
        result = make_result(exit_code=0, stdout="all good")
        context_manager.record_command_result(ts, cmd, result)
        assert ts.errors_encountered == ["some earlier failure"]

    def test_none_task_state_is_a_safe_noop(self):
        context_manager.record_command_result(None, make_cmd(), make_result())  # must not raise


class TestRecordGitResult:
    def test_records_on_success(self):
        ts = TaskState()
        context_manager.record_git_result(ts, "commit", True, "Created commit abc123: Fix bug")
        assert ts.git_operations == ["Created commit abc123: Fix bug"]

    def test_does_not_record_on_failure(self):
        ts = TaskState()
        context_manager.record_git_result(ts, "commit", False, "Failed to commit: ...")
        assert ts.git_operations == []


class TestRefreshSystemPrompt:
    def test_appends_summary_when_task_has_content(self):
        messages = [{"role": "system", "content": "Base prompt."}]
        ts = TaskState()
        ts.goal = "Add JWT authentication"
        context_manager.refresh_system_prompt(messages, ts)
        assert "Base prompt." in messages[0]["content"]
        assert "Add JWT authentication" in messages[0]["content"]

    def test_no_marker_appended_when_task_state_empty(self):
        messages = [{"role": "system", "content": "Base prompt."}]
        context_manager.refresh_system_prompt(messages, TaskState())
        assert messages[0]["content"] == "Base prompt."

    def test_regenerates_rather_than_accumulates(self):
        """Calling refresh repeatedly must not keep appending -- the summary
        section is replaced each time, not grown."""
        messages = [{"role": "system", "content": "Base prompt."}]
        ts = TaskState()
        ts.note_file_inspected("a.py")
        context_manager.refresh_system_prompt(messages, ts)
        first_length = len(messages[0]["content"])

        ts.note_file_inspected("b.py")
        context_manager.refresh_system_prompt(messages, ts)
        context_manager.refresh_system_prompt(messages, ts)
        context_manager.refresh_system_prompt(messages, ts)

        # Only one occurrence of the marker, regardless of how many refreshes.
        assert messages[0]["content"].count("Current task memory:") == 1
        assert "a.py" in messages[0]["content"]
        assert "b.py" in messages[0]["content"]

    def test_noop_when_no_system_message(self):
        messages = [{"role": "user", "content": "hi"}]
        context_manager.refresh_system_prompt(messages, TaskState())  # must not raise
        assert messages[0]["content"] == "hi"

    def test_none_task_state_clears_summary(self):
        messages = [{"role": "system", "content": "Base."}]
        ts = TaskState()
        ts.goal = "x"
        context_manager.refresh_system_prompt(messages, ts)
        assert "Current task memory" in messages[0]["content"]

        context_manager.refresh_system_prompt(messages, None)
        assert messages[0]["content"] == "Base."


def _assistant_call(name: str, args: dict) -> dict:
    return {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": name, "arguments": args}}]}


def _tool_result(name: str, content: str) -> dict:
    return {"role": "tool", "tool_name": name, "content": content}


def _cache_hit_tool_result(name: str, content: str) -> dict:
    return {"role": "tool", "tool_name": name, "content": content, "cache_hit": True}


class TestCompactMessagesDedup:
    def test_older_duplicate_read_is_superseded(self):
        messages = [
            {"role": "system", "content": "Base."},
            _assistant_call("read_file", {"path": "backend/auth.py"}),
            _tool_result("read_file", "backend/auth.py (10 lines)\n... real content ..."),
            {"role": "user", "content": "read it again"},
            _assistant_call("read_file", {"path": "backend/auth.py"}),
            _tool_result("read_file", "backend/auth.py (10 lines)\n... newer content ..."),
        ]
        context_manager.compact_messages(messages)

        assert messages[2]["content"].startswith("[earlier read_file result")
        assert "newer content" in messages[5]["content"]

    def test_different_paths_are_not_touched(self):
        messages = [
            {"role": "system", "content": "Base."},
            _assistant_call("read_file", {"path": "a.py"}),
            _tool_result("read_file", "a.py content"),
            _assistant_call("read_file", {"path": "b.py"}),
            _tool_result("read_file", "b.py content"),
        ]
        context_manager.compact_messages(messages)
        assert messages[2]["content"] == "a.py content"
        assert messages[4]["content"] == "b.py content"

    def test_most_recent_read_is_never_touched(self):
        messages = [
            {"role": "system", "content": "Base."},
            _assistant_call("read_file", {"path": "a.py"}),
            _tool_result("read_file", "first read"),
            _assistant_call("read_file", {"path": "a.py"}),
            _tool_result("read_file", "second read"),
            _assistant_call("read_file", {"path": "a.py"}),
            _tool_result("read_file", "third and latest read"),
        ]
        context_manager.compact_messages(messages)
        assert messages[6]["content"] == "third and latest read"


class TestCompactMessagesInvalidation:
    def test_read_before_edit_is_marked_stale(self):
        messages = [
            {"role": "system", "content": "Base."},
            _assistant_call("read_file", {"path": "backend/auth.py"}),
            _tool_result("read_file", "backend/auth.py (10 lines)\noriginal content"),
            _assistant_call("edit_file", {"path": "backend/auth.py", "old_text": "x", "new_text": "y"}),
            _tool_result("edit_file", "Updated 'backend/auth.py' successfully."),
        ]
        context_manager.compact_messages(messages)
        assert messages[2]["content"].startswith("[stale")
        assert "backend/auth.py" in messages[2]["content"]

    def test_write_file_also_invalidates(self):
        messages = [
            {"role": "system", "content": "Base."},
            _assistant_call("read_file", {"path": "new_file.py"}),
            _tool_result("read_file", "File not found"),
            _assistant_call("write_file", {"path": "new_file.py", "content": "x = 1"}),
            _tool_result("write_file", "Created 'new_file.py' successfully."),
        ]
        context_manager.compact_messages(messages)
        assert messages[2]["content"].startswith("[stale")

    def test_read_after_edit_is_not_invalidated(self):
        """A read that happens AFTER the edit reflects current content and
        must be left alone."""
        messages = [
            {"role": "system", "content": "Base."},
            _assistant_call("edit_file", {"path": "backend/auth.py", "old_text": "x", "new_text": "y"}),
            _tool_result("edit_file", "Updated 'backend/auth.py' successfully."),
            _assistant_call("read_file", {"path": "backend/auth.py"}),
            _tool_result("read_file", "backend/auth.py (10 lines)\nfresh content after edit"),
        ]
        context_manager.compact_messages(messages)
        assert messages[4]["content"] == "backend/auth.py (10 lines)\nfresh content after edit"


class TestCompactMessagesCacheHitAware:
    """Phase 9: a read_file result marked cache_hit (the unchanged-file
    short-circuit in tools/filesystem.py) carries no real content of its
    own, so it must never supersede -- and must never itself become -- the
    real copy of a file's content in the conversation."""

    def test_cache_hit_stub_does_not_supersede_the_real_read(self):
        messages = [
            {"role": "system", "content": "Base."},
            _assistant_call("read_file", {"path": "backend/auth.py"}),
            _tool_result("read_file", "backend/auth.py (10 lines)\n... real content ..."),
            {"role": "user", "content": "read it again"},
            _assistant_call("read_file", {"path": "backend/auth.py"}),
            _cache_hit_tool_result("read_file", "'backend/auth.py' is unchanged since you last read it."),
        ]
        context_manager.compact_messages(messages)

        # The ONLY real copy of the content must survive untouched.
        assert messages[2]["content"] == "backend/auth.py (10 lines)\n... real content ..."
        assert not messages[2]["content"].startswith("[")

    def test_a_later_real_read_still_supersedes_the_original_after_a_cache_hit(self):
        messages = [
            {"role": "system", "content": "Base."},
            _assistant_call("read_file", {"path": "a.py"}),
            _tool_result("read_file", "first real read"),
            _assistant_call("read_file", {"path": "a.py"}),
            _cache_hit_tool_result("read_file", "unchanged"),
            _assistant_call("read_file", {"path": "a.py"}),
            _tool_result("read_file", "second real read (forced)"),
        ]
        context_manager.compact_messages(messages)

        assert messages[2]["content"].startswith("[earlier read_file result")
        assert messages[6]["content"] == "second real read (forced)"


class TestCompactMessagesStats:
    def test_reports_superseded_count(self):
        messages = [
            {"role": "system", "content": "Base."},
            _assistant_call("read_file", {"path": "a.py"}),
            _tool_result("read_file", "first"),
            _assistant_call("read_file", {"path": "a.py"}),
            _tool_result("read_file", "second"),
        ]
        stats = context_manager.compact_messages(messages)
        assert stats.superseded == 1
        assert stats.stale == 0

    def test_reports_stale_count(self):
        messages = [
            {"role": "system", "content": "Base."},
            _assistant_call("read_file", {"path": "a.py"}),
            _tool_result("read_file", "original"),
            _assistant_call("edit_file", {"path": "a.py", "old_text": "x", "new_text": "y"}),
            _tool_result("edit_file", "Updated 'a.py' successfully."),
        ]
        stats = context_manager.compact_messages(messages)
        assert stats.stale == 1

    def test_reports_trimmed_count(self):
        messages = [{"role": "system", "content": "Base."}]
        for i in range(10):
            messages.append(_assistant_call("list_files", {"path": f"dir{i}"}))
            messages.append(_tool_result("list_files", "x" * 2000))
        stats = context_manager.compact_messages(messages, max_context_chars=5000, keep_recent=2)
        assert stats.trimmed > 0

    def test_empty_stats_for_a_small_untouched_conversation(self):
        messages = [
            {"role": "system", "content": "Base."},
            _assistant_call("list_files", {"path": "."}),
            _tool_result("list_files", "a.py\nb.py"),
        ]
        stats = context_manager.compact_messages(messages)
        assert stats.superseded == 0
        assert stats.stale == 0
        assert stats.trimmed == 0
        assert bool(stats) is False


class TestCompactMessagesSizeFallback:
    def test_old_large_tool_output_trimmed_when_over_budget(self):
        messages = [{"role": "system", "content": "Base."}]
        # 10 unrelated, large tool results -- no path-based dedup applies.
        for i in range(10):
            messages.append(_assistant_call("list_files", {"path": f"dir{i}"}))
            messages.append(_tool_result("list_files", "x" * 2000))

        context_manager.compact_messages(messages, max_context_chars=5000, keep_recent=2)

        total = sum(len(m.get("content") or "") for m in messages)
        assert total <= 5000 + len(context_manager._TRIMMED_PLACEHOLDER)

    def test_recent_tool_messages_are_protected(self):
        messages = [{"role": "system", "content": "Base."}]
        for i in range(10):
            messages.append(_assistant_call("list_files", {"path": f"dir{i}"}))
            messages.append(_tool_result("list_files", "x" * 2000))

        context_manager.compact_messages(messages, max_context_chars=1, keep_recent=2)

        tool_messages = [m for m in messages if m.get("role") == "tool"]
        # The last 2 must survive untrimmed even though the budget is tiny.
        assert tool_messages[-1]["content"] == "x" * 2000
        assert tool_messages[-2]["content"] == "x" * 2000
        # At least one older one must have been trimmed.
        assert any(m["content"].startswith("[tool output omitted") for m in tool_messages[:-2])

    def test_small_conversation_untouched(self):
        messages = [
            {"role": "system", "content": "Base."},
            _assistant_call("list_files", {"path": "."}),
            _tool_result("list_files", "a.py\nb.py"),
        ]
        context_manager.compact_messages(messages)
        assert messages[2]["content"] == "a.py\nb.py"

    def test_does_not_touch_system_or_user_messages(self):
        messages = [{"role": "system", "content": "Base."}]
        for i in range(10):
            messages.append({"role": "user", "content": "x" * 2000})
            messages.append(_assistant_call("list_files", {"path": f"dir{i}"}))
            messages.append(_tool_result("list_files", "y" * 2000))

        context_manager.compact_messages(messages, max_context_chars=1000, keep_recent=1)

        user_messages = [m for m in messages if m.get("role") == "user"]
        assert all(m["content"] == "x" * 2000 for m in user_messages)
